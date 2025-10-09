"""
Matcher Unificado para Análisis Balístico
Sistema Balístico Forense MVP

Módulo consolidado para matching y comparación de características balísticas
que combina las funcionalidades de matcher.py, improved_matcher.py, optimized_matcher.py
y alternative_matchers.py

Implementa y evalúa múltiples algoritmos:
- ORB (Oriented FAST and Rotated BRIEF)
- SIFT (Scale-Invariant Feature Transform)
- AKAZE (Accelerated-KAZE)
- BRISK (Binary Robust Invariant Scalable Keypoints)
- KAZE (Non-linear diffusion filtering)

Basado en literatura científica:
- Rublee et al. (2011) - ORB
- Lowe (2004) - SIFT
- Alcantarilla et al. (2013) - KAZE/AKAZE
- Leutenegger et al. (2011) - BRISK
- Evaluación específica para dominio balístico
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
import json
from pathlib import Path
import os

from utils.logger import LoggerMixin
from matching.cmc_algorithm import CMCAlgorithm, CMCParameters, CMCMatchResult
from image_processing.ballistic_features import BallisticFeatureExtractor

# Importar módulo unificado de funciones de similitud
try:
    from common.similarity_functions_unified import (
        UnifiedSimilarityAnalyzer, SimilarityConfig, SimilarityBootstrapResult
    )
    UNIFIED_SIMILARITY_AVAILABLE = True
except ImportError:
    UNIFIED_SIMILARITY_AVAILABLE = False
    logging.warning("Módulo unificado de similitud no disponible, usando implementación legacy")

# GPU acceleration imports
try:
    from image_processing.gpu_accelerator import GPUAccelerator
except ImportError:
    class GPUAccelerator:
        def __init__(self):
            self.enabled = False
        def is_gpu_enabled(self):
            return False

class AlgorithmType(Enum):
    """Tipos de algoritmos de detección de características"""
    ORB = "ORB"
    SIFT = "SIFT"
    AKAZE = "AKAZE"
    BRISK = "BRISK"
    KAZE = "KAZE"
    CMC = "CMC"

class MatchingLevel(Enum):
    """Niveles de matching"""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"

@dataclass
class MatchResult:
    """Resultado de un matching entre dos imágenes"""
    # Información básica
    image1_id: int = 0
    image2_id: int = 0
    algorithm: str = "ORB"
    
    # Estadísticas de keypoints y matches
    total_keypoints1: int = 0
    total_keypoints2: int = 0
    total_matches: int = 0
    good_matches: int = 0
    
    # Métricas de similitud
    similarity_score: float = 0.0
    confidence: float = 0.0
    
    # Bootstrap confidence intervals
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    confidence_interval_method: str = "basic"  # 'percentile', 'basic', 'bca'
    bootstrap_samples: int = 0
    bootstrap_confidence_level: float = 0.95
    bootstrap_bias: float = 0.0
    bootstrap_std_error: float = 0.0
    bootstrap_used: bool = False
    
    # Datos adicionales
    match_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    # Métricas avanzadas (opcionales)
    geometric_consistency: float = 0.0
    match_quality: float = 0.0
    keypoint_density: float = 0.0
    descriptor_distance_stats: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics para similarity weighting
    image1_quality_score: float = 0.0
    image2_quality_score: float = 0.0
    combined_quality_score: float = 0.0
    quality_weighted_similarity: float = 0.0

@dataclass
class KeyPoint:
    """Punto clave para visualización"""
    x: float
    y: float
    size: float
    angle: float
    response: float

@dataclass
class Match:
    """Match entre dos puntos clave"""
    kp1: KeyPoint
    kp2: KeyPoint
    distance: float
    confidence: float

@dataclass
class MatchingConfig:
    """Configuración para matching"""
    # Configuración general
    level: MatchingLevel = MatchingLevel.STANDARD
    algorithm: AlgorithmType = AlgorithmType.ORB
    
    # Parámetros de extracción de características
    max_features: int = 1000
    
    # Parámetros de matching
    lowe_ratio: float = 0.7
    min_matches_for_ransac: int = 10
    ransac_threshold: float = 5.0
    
    # Parámetros de filtrado
    distance_threshold: float = 0.75
    min_matches: int = 10
    similarity_threshold: float = 0.3
    
    # Parámetros de visualización
    max_matches_to_draw: int = 50
    
    # Parámetros de aceleración GPU
    enable_gpu_acceleration: bool = True
    gpu_device_id: int = 0
    gpu_fallback_to_cpu: bool = True

class UnifiedMatcher(LoggerMixin):
    """Matcher unificado para imágenes balísticas"""
    
    def __init__(self, config: Optional[Union[Dict, MatchingConfig]] = None):
        """
        Inicializa el matcher
        
        Args:
            config: Configuración del matcher (opcional)
                   Puede ser un diccionario o un objeto MatchingConfig
        """
        super().__init__()
        
        # Configuraciones predefinidas
        self.default_configs = {
            MatchingLevel.BASIC: MatchingConfig(
                level=MatchingLevel.BASIC,
                algorithm=AlgorithmType.ORB,
                max_features=500,
                lowe_ratio=0.7,
                min_matches_for_ransac=10,
                ransac_threshold=5.0,
                distance_threshold=0.75,
                min_matches=10,
                similarity_threshold=0.3,
                max_matches_to_draw=50
            ),
            MatchingLevel.STANDARD: MatchingConfig(
                level=MatchingLevel.STANDARD,
                algorithm=AlgorithmType.ORB,
                max_features=1000,
                lowe_ratio=0.75,
                min_matches_for_ransac=10,
                ransac_threshold=5.0,
                distance_threshold=0.75,
                min_matches=10,
                similarity_threshold=0.3,
                max_matches_to_draw=50
            ),
            MatchingLevel.ADVANCED: MatchingConfig(
                level=MatchingLevel.ADVANCED,
                algorithm=AlgorithmType.SIFT,
                max_features=2000,
                lowe_ratio=0.8,
                min_matches_for_ransac=8,
                ransac_threshold=4.0,
                distance_threshold=0.7,
                min_matches=8,
                similarity_threshold=0.25,
                max_matches_to_draw=100
            )
        }
        
        # Procesar configuración
        if config is None:
            self.config = self.default_configs[MatchingLevel.STANDARD]
        elif isinstance(config, dict):
            # Convertir diccionario a MatchingConfig
            level_str = config.get('level', 'standard').lower()
            level = next((l for l in MatchingLevel if l.value == level_str), 
                         MatchingLevel.STANDARD)
            
            # Partir de la configuración predeterminada y actualizar
            self.config = self.default_configs[level]
            
            # Actualizar con valores del diccionario
            for key, value in config.items():
                if hasattr(self.config, key):
                    # Manejar conversión especial para campos enum
                    if key == 'algorithm' and isinstance(value, str):
                        # Convertir string a AlgorithmType enum
                        algorithm_value = next((a for a in AlgorithmType if a.value == value), 
                                             AlgorithmType.ORB)
                        setattr(self.config, key, algorithm_value)
                    elif key == 'level' and isinstance(value, str):
                        # Convertir string a MatchingLevel enum
                        level_value = next((l for l in MatchingLevel if l.value == value), 
                                         MatchingLevel.STANDARD)
                        setattr(self.config, key, level_value)
                    else:
                        setattr(self.config, key, value)
        else:
            # Usar configuración proporcionada directamente
            self.config = config
        
        # Inicializar detectores de características
        self.feature_detectors = self._initialize_detectors()
        
        # Inicializar matchers
        self.matchers = self._initialize_matchers()
        
        # Inicializar algoritmo CMC
        self.cmc_algorithm = CMCAlgorithm()
        
        # Inicializar extractor de características balísticas para quality scoring
        self.ballistic_extractor = BallisticFeatureExtractor()
        
        # Inicializar aceleración GPU
        self.gpu_accelerator = None
        if self.config.enable_gpu_acceleration:
            try:
                self.gpu_accelerator = GPUAccelerator(device_id=self.config.gpu_device_id)
                if self.gpu_accelerator.is_gpu_enabled():
                    self.logger.info(f"GPU acceleration enabled: {self.gpu_accelerator.get_gpu_info()}")
                else:
                    self.logger.warning("GPU acceleration requested but not available")
                    if not self.config.gpu_fallback_to_cpu:
                        raise RuntimeError("GPU acceleration required but not available")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU acceleration: {e}")
                if not self.config.gpu_fallback_to_cpu:
                    raise
                self.gpu_accelerator = None
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            'total_comparisons': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'average_processing_time': 0.0,
            'quality_weighted_improvements': 0
        }
        
        self.logger.info(f"Matcher unificado inicializado (nivel: {self.config.level.value}, algoritmo: {self.config.algorithm.value})")
    
    def _initialize_detectors(self) -> Dict[AlgorithmType, Any]:
        """
        Inicializa los detectores de características
        
        Returns:
            Diccionario con detectores de características
        """
        detectors = {}
        
        # ORB
        detectors[AlgorithmType.ORB] = cv2.ORB_create(
            nfeatures=self.config.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # SIFT (si está disponible)
        try:
            detectors[AlgorithmType.SIFT] = cv2.SIFT_create(
                nfeatures=self.config.max_features,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        except AttributeError:
            self.logger.warning("SIFT no disponible en esta versión de OpenCV")
        
        # AKAZE
        try:
            detectors[AlgorithmType.AKAZE] = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4
            )
        except AttributeError:
            self.logger.warning("AKAZE no disponible en esta versión de OpenCV")
        
        # BRISK
        try:
            detectors[AlgorithmType.BRISK] = cv2.BRISK_create(
                thresh=30,
                octaves=3,
                patternScale=1.0
            )
        except AttributeError:
            self.logger.warning("BRISK no disponible en esta versión de OpenCV")
        
        # KAZE
        try:
            detectors[AlgorithmType.KAZE] = cv2.KAZE_create(
                extended=False,
                upright=False,
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4
            )
        except AttributeError:
            self.logger.warning("KAZE no disponible en esta versión de OpenCV")
        
        return detectors
    
    def _initialize_matchers(self) -> Dict[AlgorithmType, Any]:
        """
        Inicializa los matchers
        
        Returns:
            Diccionario con matchers
        """
        matchers = {}
        
        # Matcher para ORB y BRISK (descriptores binarios)
        matchers[AlgorithmType.ORB] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matchers[AlgorithmType.BRISK] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Matcher para SIFT, KAZE y AKAZE (descriptores de punto flotante)
        matchers[AlgorithmType.SIFT] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matchers[AlgorithmType.KAZE] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matchers[AlgorithmType.AKAZE] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Configurar FLANN matcher como alternativa
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        return matchers
    
    def extract_features(self, image: np.ndarray, 
                        algorithm: Optional[Union[AlgorithmType, str]] = None) -> Dict[str, Any]:
        """
        Extrae características de una imagen
        
        Args:
            image: Imagen a procesar
            algorithm: Algoritmo a utilizar (opcional)
            
        Returns:
            Diccionario con características extraídas
        """
        try:
            # Usar algoritmo especificado o el de la configuración
            if isinstance(algorithm, str):
                # Convertir string a AlgorithmType
                try:
                    alg = AlgorithmType(algorithm)
                except ValueError:
                    self.logger.warning(f"Algoritmo '{algorithm}' no válido. Usando ORB.")
                    alg = AlgorithmType.ORB
            else:
                alg = algorithm or self.config.algorithm
            
            # Caso especial para CMC
            if alg == AlgorithmType.CMC:
                # Para CMC, solo preprocesamos la imagen
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image.copy()
                
                processed = self._preprocess_image(gray)
                
                return {
                    "algorithm": alg.value,
                    "image": processed,
                    "keypoints": [],
                    "descriptors": None,
                    "num_keypoints": 0,
                    "keypoint_density": 0.0,
                    "image_shape": processed.shape,
                    "original_image": image  # Incluir imagen original para cálculo de quality score
                }
            
            # Verificar que el algoritmo está disponible
            if alg not in self.feature_detectors:
                self.logger.warning(f"Algoritmo {alg.value} no disponible. Usando ORB.")
                alg = AlgorithmType.ORB
            
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Preprocesar imagen
            processed = self._preprocess_image(gray)
            
            # Extraer características usando GPU si está disponible
            keypoints = None
            descriptors = None
            
            if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
                try:
                    if alg == AlgorithmType.ORB:
                        keypoints, descriptors = self.gpu_accelerator.extract_orb_features(
                            processed, self.config.max_features)
                    elif alg == AlgorithmType.SIFT:
                        keypoints, descriptors = self.gpu_accelerator.extract_sift_features(
                            processed, self.config.max_features)
                    elif alg == AlgorithmType.AKAZE:
                        keypoints, descriptors = self.gpu_accelerator.extract_akaze_features(processed)
                    elif alg == AlgorithmType.BRISK:
                        keypoints, descriptors = self.gpu_accelerator.extract_brisk_features(processed)
                    elif alg == AlgorithmType.KAZE:
                        keypoints, descriptors = self.gpu_accelerator.extract_kaze_features(processed)
                    
                    if keypoints is not None and descriptors is not None:
                        self.logger.debug(f"Características extraídas con GPU usando {alg.value}")
                    else:
                        raise Exception("GPU feature extraction returned None")
                        
                except Exception as e:
                    self.logger.warning(f"Error en extracción GPU con {alg.value}: {e}")
                    keypoints = None
                    descriptors = None
            
            # Fallback a CPU si GPU falló o no está disponible
            if keypoints is None or descriptors is None:
                keypoints, descriptors = self.feature_detectors[alg].detectAndCompute(processed, None)
            
            # Verificar resultados
            if keypoints is None or len(keypoints) == 0:
                self.logger.warning(f"No se detectaron keypoints con {alg.value}")
                return {
                    "algorithm": alg.value,
                    "keypoints": [],
                    "descriptors": None,
                    "num_keypoints": 0
                }
            
            if descriptors is None:
                self.logger.warning(f"No se pudieron extraer descriptores con {alg.value}")
                return {
                    "algorithm": alg.value,
                    "keypoints": keypoints,
                    "descriptors": None,
                    "num_keypoints": len(keypoints)
                }
            
            # Calcular densidad de keypoints
            height, width = processed.shape
            density = len(keypoints) / (height * width)
            
            return {
                "algorithm": alg.value,
                "keypoints": keypoints,
                "descriptors": descriptors,
                "num_keypoints": len(keypoints),
                "keypoint_density": density,
                "image_shape": processed.shape,
                "original_image": image  # Incluir imagen original para cálculo de quality score
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características: {e}")
            return {
                "algorithm": alg.value if 'alg' in locals() else self.config.algorithm.value,
                "keypoints": [],
                "descriptors": None,
                "num_keypoints": 0,
                "error": str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para extracción de características con aceleración GPU
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Imagen preprocesada
        """
        try:
            # Aplicar filtro gaussiano para reducir ruido
            if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
                blurred = self.gpu_accelerator.gaussian_blur(image, (3, 3), 0)
                # Normalizar contraste usando GPU
                enhanced = self.gpu_accelerator.clahe(blurred, clip_limit=2.0, tile_grid_size=(8, 8))
            else:
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                # Normalizar contraste
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(blurred)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {e}")
            return image
    
    def match_features(self, features1: Dict[str, Any], features2: Dict[str, Any],
                      algorithm: Optional[AlgorithmType] = None) -> MatchResult:
        """
        Realiza matching entre dos conjuntos de características
        
        Args:
            features1: Características de la primera imagen
            features2: Características de la segunda imagen
            algorithm: Algoritmo a utilizar (opcional)
            
        Returns:
            Resultado del matching
        """
        start_time = time.time()
        
        try:
            # Usar algoritmo especificado o el de las características
            alg_str = features1.get("algorithm", self.config.algorithm.value)
            alg = next((a for a in AlgorithmType if a.value == alg_str), self.config.algorithm)
            
            # Caso especial para CMC
            if alg == AlgorithmType.CMC:
                return self._match_cmc(features1, features2)
            
            # Verificar que hay características
            if (features1.get("descriptors") is None or 
                features2.get("descriptors") is None or 
                features1.get("num_keypoints", 0) == 0 or 
                features2.get("num_keypoints", 0) == 0):
                
                return MatchResult(
                    algorithm=features1.get("algorithm", self.config.algorithm.value),
                    total_keypoints1=features1.get("num_keypoints", 0),
                    total_keypoints2=features2.get("num_keypoints", 0),
                    processing_time=time.time() - start_time,
                    similarity_score=0.0  # Asegurar que similarity_score no sea None
                )
            
            # Verificar que el algoritmo está disponible
            if alg not in self.matchers:
                self.logger.warning(f"Matcher para {alg.value} no disponible. Usando ORB.")
                alg = AlgorithmType.ORB
            
            # Obtener descriptores y keypoints
            desc1 = features1["descriptors"]
            desc2 = features2["descriptors"]
            kp1 = features1["keypoints"]
            kp2 = features2["keypoints"]
            
            # Realizar matching según el algoritmo
            if alg in [AlgorithmType.SIFT, AlgorithmType.KAZE]:
                # Usar knnMatch para algoritmos de punto flotante
                if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
                    try:
                        matches = self.gpu_accelerator.match_descriptors(desc1, desc2, method='knn', k=2)
                    except Exception as e:
                        self.logger.warning(f"GPU matching failed, falling back to CPU: {e}")
                        matches = self.matchers[alg].knnMatch(desc1, desc2, k=2)
                else:
                    matches = self.matchers[alg].knnMatch(desc1, desc2, k=2)
                
                # Aplicar Lowe's ratio test
                good_matches = []
                all_matches = []
                
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        all_matches.append(m)
                        if m.distance < self.config.lowe_ratio * n.distance:
                            good_matches.append(m)
                
                # Calcular distancias
                distances = [m.distance for m in all_matches] if all_matches else []
                
            else:
                # Usar match para algoritmos binarios
                if self.gpu_accelerator and self.gpu_accelerator.is_gpu_enabled():
                    try:
                        matches = self.gpu_accelerator.match_descriptors(desc1, desc2, method='bf')
                    except Exception as e:
                        self.logger.warning(f"GPU matching failed, falling back to CPU: {e}")
                        matches = self.matchers[alg].match(desc1, desc2)
                else:
                    matches = self.matchers[alg].match(desc1, desc2)
                
                # Ordenar por distancia
                matches = sorted(matches, key=lambda x: x.distance)
                all_matches = matches
                
                # Filtrar matches buenos (usando threshold adaptativo mejorado)
                if matches:
                    distances = [m.distance for m in matches]
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    
                    # Manejar caso especial de auto-match (distancias muy bajas)
                    if mean_dist < 5.0:  # Distancias muy bajas, probablemente auto-match
                        # Para auto-match, usar un threshold más permisivo
                        threshold = max(mean_dist + std_dist, 10.0)
                    else:
                        # Threshold adaptativo normal
                        threshold = min(mean_dist - 0.3 * std_dist, self.config.distance_threshold * 100)
                    
                    # Asegurar que el threshold no sea negativo
                    threshold = max(threshold, 0.1)
                    
                    good_matches = [m for m in matches if m.distance <= threshold]
                    
                    # Si no hay good matches con threshold adaptativo, usar los mejores N matches
                    if len(good_matches) == 0 and len(matches) > 0:
                        # Tomar el mejor 30% de los matches o mínimo 10
                        num_good = max(min(len(matches) // 3, 50), min(10, len(matches)))
                        good_matches = matches[:num_good]
                else:
                    distances = []
                    good_matches = []
            
            # Calcular estadísticas de distancias
            distance_stats = {
                "mean": float(np.mean(distances)) if distances else 0.0,
                "std": float(np.std(distances)) if distances else 0.0,
                "min": float(np.min(distances)) if distances else 0.0,
                "max": float(np.max(distances)) if distances else 0.0
            }
            
            # Calcular quality scores de las imágenes usando el extractor balístico
            image1_quality = 0.0
            image2_quality = 0.0
            
            try:
                # Obtener las imágenes originales desde las características
                if "original_image" in features1 and "original_image" in features2:
                    img1 = features1["original_image"]
                    img2 = features2["original_image"]
                    
                    # Calcular quality scores usando el extractor balístico
                    image1_quality = self.ballistic_extractor._calculate_quality_score(img1, [])
                    image2_quality = self.ballistic_extractor._calculate_quality_score(img2, [])
                    
                    self.logger.debug(f"Quality scores calculados: img1={image1_quality:.3f}, img2={image2_quality:.3f}")
                else:
                    self.logger.warning("Imágenes originales no disponibles para cálculo de quality score")
            except Exception as e:
                self.logger.warning(f"Error calculando quality scores: {e}")
            
            # Calcular score de similitud con quality weighting
            base_score, combined_quality, quality_weighted_score = self._calculate_similarity_score(
                len(good_matches), len(kp1), len(kp2), alg.value, image1_quality, image2_quality
            )
            
            # Calcular confianza con información bootstrap
            confidence, bootstrap_info = self._calculate_confidence(good_matches, all_matches, alg.value)
            
            # Calcular consistencia geométrica si hay suficientes matches
            geometric_consistency = 0.0
            if len(good_matches) >= self.config.min_matches_for_ransac:
                geometric_consistency = self._calculate_geometric_consistency(
                    kp1, kp2, good_matches
                )
            
            # Calcular calidad de matches
            match_quality = self._calculate_match_quality(
                [m.distance for m in good_matches] if good_matches else []
            )
            
            # Preparar datos del match
            match_data = {
                "matches": [
                    {
                        "kp1_idx": m.queryIdx,
                        "kp2_idx": m.trainIdx,
                        "distance": float(m.distance)
                    } for m in good_matches[:50]  # Limitar para evitar datos excesivos
                ],
                "distance_stats": distance_stats,
                "geometric_consistency": geometric_consistency,
                "match_quality": match_quality
            }
            
            # Si se usó Lowe's ratio test, incluir el ratio
            if alg in [AlgorithmType.SIFT, AlgorithmType.KAZE]:
                match_data["lowe_ratio"] = self.config.lowe_ratio
            
            processing_time = time.time() - start_time
            
            # Crear resultado
            result = MatchResult(
                algorithm=alg.value,
                total_keypoints1=len(kp1),
                total_keypoints2=len(kp2),
                total_matches=len(all_matches),
                good_matches=len(good_matches),
                similarity_score=quality_weighted_score,  # Usar el score ponderado por calidad
                confidence=confidence,
                # Campos bootstrap
                confidence_interval_lower=bootstrap_info.get("confidence_interval_lower", 0.0),
                confidence_interval_upper=bootstrap_info.get("confidence_interval_upper", 0.0),
                confidence_interval_method=bootstrap_info.get("confidence_interval_method", "basic"),
                bootstrap_samples=bootstrap_info.get("bootstrap_samples", 0),
                bootstrap_confidence_level=bootstrap_info.get("bootstrap_confidence_level", 0.95),
                bootstrap_bias=bootstrap_info.get("bootstrap_bias", 0.0),
                bootstrap_std_error=bootstrap_info.get("bootstrap_std_error", 0.0),
                bootstrap_used=bootstrap_info.get("bootstrap_used", False),
                match_data=match_data,
                processing_time=processing_time,
                geometric_consistency=geometric_consistency,
                match_quality=match_quality,
                keypoint_density=(features1.get("keypoint_density", 0) + 
                                 features2.get("keypoint_density", 0)) / 2,
                descriptor_distance_stats=distance_stats,
                # Nuevos campos de quality weighting
                image1_quality_score=image1_quality,
                image2_quality_score=image2_quality,
                combined_quality_score=combined_quality,
                quality_weighted_similarity=quality_weighted_score
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en matching: {e}")
            
            # Crear resultado de error
            return MatchResult(
                error_message=f"Error comparando imágenes: {e}",
                similarity_score=0.0
            )
    
    def _match_cmc(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> MatchResult:
        """
        Realiza matching usando el algoritmo CMC (Congruent Matching Cells).
        
        Args:
            features1: Características de la primera imagen (contiene la imagen preprocesada)
            features2: Características de la segunda imagen (contiene la imagen preprocesada)
            
        Returns:
            MatchResult: Resultado del matching CMC
        """
        start_time = time.time()
        
        try:
            # Obtener las imágenes preprocesadas
            img1 = features1.get("image")
            img2 = features2.get("image")
            
            if img1 is None or img2 is None:
                raise ValueError("Las imágenes preprocesadas no están disponibles para CMC")
            
            # Realizar comparación CMC bidireccional
            cmc_result = self.cmc_algorithm.compare_images_bidirectional(img1, img2)
            
            # Calcular métricas de similitud basadas en CMC
            similarity_score = self._calculate_cmc_similarity_score(cmc_result)
            confidence = self._calculate_cmc_confidence(cmc_result)
            
            # Preparar datos del match específicos para CMC
            match_data = {
                "cmc_score": cmc_result.cmc_score,
                "cmc_count": cmc_result.cmc_count,
                "total_cells": cmc_result.total_cells,
                "valid_cells": cmc_result.valid_cells,
                "convergence_score": cmc_result.convergence_score,
                "is_match": cmc_result.is_match,
                "num_cells_x": self.cmc_algorithm.parameters.num_cells_x,
                "num_cells_y": self.cmc_algorithm.parameters.num_cells_y,
                "ccf_threshold": self.cmc_algorithm.parameters.ccf_threshold
            }
            
            processing_time = time.time() - start_time
            
            # Crear resultado CMC
            result = MatchResult(
                algorithm="CMC",
                total_keypoints1=0,  # CMC no usa keypoints tradicionales
                total_keypoints2=0,
                total_matches=cmc_result.total_cells,
                good_matches=cmc_result.cmc_count,
                similarity_score=similarity_score,
                confidence=confidence,
                match_data=match_data,
                processing_time=processing_time,
                geometric_consistency=cmc_result.convergence_score or 0.0,
                match_quality=confidence,
                keypoint_density=0.0,  # No aplica para CMC
                descriptor_distance_stats={}  # No aplica para CMC
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en matching CMC: {e}")
            
            # Crear resultado de error
            return MatchResult(
                algorithm="CMC",
                total_keypoints1=0,
                total_keypoints2=0,
                processing_time=time.time() - start_time,
                match_data={"error": str(e)},
                similarity_score=0.0  # Asegurar que similarity_score no sea None en caso de error
            )
    
    def _calculate_cmc_similarity_score(self, cmc_result) -> float:
        """
        Calcula el score de similitud basado en el resultado CMC.
        
        Args:
            cmc_result: Resultado del algoritmo CMC (CMCMatchResult)
            
        Returns:
            float: Score de similitud entre 0 y 1
        """
        if cmc_result.total_cells == 0:
            return 0.0
        
        # Score basado en la proporción de celdas CMC
        base_score = cmc_result.cmc_count / cmc_result.total_cells
        
        # Ajustar por convergencia si está disponible
        convergence_factor = cmc_result.convergence_score or 0.0
        
        # Score final combinando ambos factores
        final_score = (base_score * 0.7) + (convergence_factor * 0.3)
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_cmc_confidence(self, cmc_result) -> float:
        """
        Calcula la confianza basada en el resultado CMC.
        
        Args:
            cmc_result: Resultado del algoritmo CMC (CMCMatchResult)
            
        Returns:
            float: Confianza entre 0 y 1
        """
        # Confianza basada en el número de celdas CMC y convergencia
        if cmc_result.total_cells == 0:
            return 0.0
        
        # Factor basado en el número absoluto de celdas CMC
        cell_factor = min(cmc_result.cmc_count / 50.0, 1.0)  # Normalizar a 50 celdas
        
        # Factor basado en la proporción de celdas CMC
        ratio_factor = cmc_result.cmc_count / cmc_result.total_cells
        
        # Factor de convergencia bidireccional
        convergence_factor = cmc_result.convergence_score or 0.0
        
        # Confianza final
        confidence = (cell_factor * 0.3) + (ratio_factor * 0.4) + (convergence_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def _calculate_similarity_score(self, good_matches: int, kp1_count: int, 
                                  kp2_count: int, algorithm: str, 
                                  image1_quality: float = 0.0, 
                                  image2_quality: float = 0.0) -> Tuple[float, float, float]:
        """
        Calcula el score de similitud basado en matches, keypoints y quality weighting
        
        Args:
            good_matches: Número de buenos matches
            kp1_count: Número de keypoints en la primera imagen
            kp2_count: Número de keypoints en la segunda imagen
            algorithm: Algoritmo utilizado
            image1_quality: Quality score de la primera imagen (0-1)
            image2_quality: Quality score de la segunda imagen (0-1)
            
        Returns:
            Tuple[base_score, combined_quality, quality_weighted_score]: 
            - Score base de similitud (0-100)
            - Quality score combinado (0-1)
            - Score ponderado por calidad (0-100)
        """
        if kp1_count == 0 or kp2_count == 0:
            return 0.0, 0.0, 0.0
        
        # Score basado en la proporción de matches buenos
        max_possible_matches = min(kp1_count, kp2_count)
        if max_possible_matches == 0:
            return 0.0, 0.0, 0.0
        
        # Score base: proporción de matches buenos
        base_score = good_matches / max_possible_matches
        
        # Ajustar según el algoritmo
        if algorithm == AlgorithmType.ORB.value:
            # ORB tiende a tener menos matches pero más precisos
            if good_matches >= 10:
                base_score *= 1.2  # Bonus por tener suficientes matches
        elif algorithm == AlgorithmType.SIFT.value:
            # SIFT puede tener más matches, ajustar accordingly
            if good_matches >= 20:
                base_score *= 1.1
        elif algorithm == AlgorithmType.AKAZE.value:
            # AKAZE es más selectivo
            if good_matches >= 8:
                base_score *= 1.15
        
        # Normalizar a 0-100
        base_score = min(base_score * 100, 100.0)
        
        # Calcular quality score combinado
        # Usar media armónica para penalizar imágenes de baja calidad
        if image1_quality > 0 and image2_quality > 0:
            combined_quality = 2 * (image1_quality * image2_quality) / (image1_quality + image2_quality)
        else:
            combined_quality = (image1_quality + image2_quality) / 2
        
        # Aplicar quality weighting al similarity score
        # Fórmula: score_weighted = base_score * (0.5 + 0.5 * combined_quality)
        # Esto asegura que incluso con calidad 0, el score no sea completamente 0
        quality_factor = 0.5 + 0.5 * combined_quality
        quality_weighted_score = base_score * quality_factor
        
        # Log para debugging
        self.logger.debug(f"Similarity calculation - Base: {base_score:.2f}, "
                         f"Quality1: {image1_quality:.3f}, Quality2: {image2_quality:.3f}, "
                         f"Combined Quality: {combined_quality:.3f}, "
                         f"Quality Factor: {quality_factor:.3f}, "
                         f"Weighted Score: {quality_weighted_score:.2f}")
        
        return base_score, combined_quality, quality_weighted_score
    
    def _calculate_confidence(self, good_matches: List, all_matches: List, 
                            algorithm: str = "ORB", use_bootstrap: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Calcula la confianza del matching con opción de bootstrap sampling usando módulo unificado
        
        Args:
            good_matches: Lista de buenos matches
            all_matches: Lista de todos los matches
            algorithm: Algoritmo utilizado para ajustes específicos
            use_bootstrap: Si usar bootstrap para intervalos de confianza
            
        Returns:
            Tuple[float, Dict]: (confianza, información_bootstrap)
        """
        bootstrap_info = {
            'confidence_interval_lower': 0.0,
            'confidence_interval_upper': 0.0,
            'confidence_interval_method': 'basic',
            'bootstrap_samples': 0,
            'bootstrap_confidence_level': 0.95,
            'bootstrap_bias': 0.0,
            'bootstrap_std_error': 0.0,
            'bootstrap_used': False
        }
        
        if not all_matches:
            return 0.0, bootstrap_info
        
        # Confianza básica basada en la proporción de matches buenos
        ratio = len(good_matches) / len(all_matches)
        
        # Ajustar por cantidad absoluta de matches
        if len(good_matches) >= 15:
            base_confidence = ratio * 0.9 + 0.1  # Alta confianza
        elif len(good_matches) >= 8:
            base_confidence = ratio * 0.7 + 0.1  # Confianza media
        else:
            base_confidence = ratio * 0.5  # Baja confianza
        
        base_confidence = min(base_confidence * 100, 100.0)
        
        # Si no se usa bootstrap o hay pocos matches, retornar confianza básica
        if not use_bootstrap or len(good_matches) < 5:
            return base_confidence, bootstrap_info
        
        # Intentar usar módulo unificado de similitud
        if UNIFIED_SIMILARITY_AVAILABLE:
            try:
                # Crear configuración para análisis de similitud
                config = SimilarityConfig(
                    n_bootstrap=500,
                    confidence_level=0.95,
                    method='percentile',
                    parallel=True
                )
                
                # Crear analizador unificado
                analyzer = UnifiedSimilarityAnalyzer(config)
                
                # Convertir matches a formato compatible
                matches_data = []
                for match in good_matches:
                    if hasattr(match, 'distance'):
                        matches_data.append({
                            'distance': match.distance,
                            'kp1_idx': getattr(match, 'queryIdx', 0),
                            'kp2_idx': getattr(match, 'trainIdx', 0)
                        })
                    else:
                        matches_data.append({
                            'distance': match.get('distance', 0.5),
                            'kp1_idx': match.get('kp1_idx', 0),
                            'kp2_idx': match.get('kp2_idx', 0)
                        })
                
                if matches_data:
                    # Calcular bootstrap con módulo unificado
                    ci_lower, ci_upper, bootstrap_info_result = analyzer.analyze_bootstrap_confidence_interval(
                        matches_data, base_confidence, algorithm
                    )
                    
                    # Actualizar información bootstrap
                    bootstrap_info.update(bootstrap_info_result)
                    
                    # Usar el ancho del intervalo para ajustar la confianza
                    ci_width = ci_upper - ci_lower
                    max_expected_width = 30.0
                    width_factor = max(0.8, min(1.2, 1.0 - (ci_width / max_expected_width) * 0.2))
                    bootstrap_confidence = base_confidence * width_factor
                    
                    self.logger.debug(f"Unified bootstrap confidence - "
                                    f"Base: {base_confidence:.2f}, "
                                    f"CI: [{ci_lower:.2f}, {ci_upper:.2f}], "
                                    f"Final: {bootstrap_confidence:.2f}")
                    
                    return min(bootstrap_confidence, 100.0), bootstrap_info
                    
            except Exception as e:
                self.logger.warning(f"Error usando módulo unificado de similitud: {e}")
        
        # Fallback a implementación legacy
        try:
            from matching.bootstrap_similarity import calculate_bootstrap_confidence_interval
            
            # Convertir matches a formato compatible con bootstrap
            matches_data = []
            for match in good_matches:
                if hasattr(match, 'distance'):
                    matches_data.append({
                        'distance': match.distance,
                        'kp1_idx': getattr(match, 'queryIdx', 0),
                        'kp2_idx': getattr(match, 'trainIdx', 0)
                    })
                else:
                    # Fallback para matches en formato dict
                    matches_data.append({
                        'distance': match.get('distance', 0.5),
                        'kp1_idx': match.get('kp1_idx', 0),
                        'kp2_idx': match.get('kp2_idx', 0)
                    })
            
            if matches_data:
                # Calcular intervalo de confianza bootstrap
                ci_lower, ci_upper = calculate_bootstrap_confidence_interval(
                    matches_data, 
                    base_confidence,
                    algorithm=algorithm,
                    confidence_level=0.95,
                    n_bootstrap=500  # Número moderado para balance velocidad/precisión
                )
                
                # Actualizar información bootstrap
                bootstrap_info.update({
                    'confidence_interval_lower': ci_lower,
                    'confidence_interval_upper': ci_upper,
                    'confidence_interval_method': 'percentile',
                    'bootstrap_samples': 500,
                    'bootstrap_confidence_level': 0.95,
                    'bootstrap_bias': 0.0,  # Simplificado para esta implementación
                    'bootstrap_std_error': (ci_upper - ci_lower) / 3.92,  # Aproximación
                    'bootstrap_used': True
                })
                
                # Usar el ancho del intervalo para ajustar la confianza
                # Intervalos más estrechos indican mayor confianza
                ci_width = ci_upper - ci_lower
                max_expected_width = 30.0  # Ancho máximo esperado para CI
                
                # Factor de ajuste basado en ancho del CI (0.8 a 1.2)
                width_factor = max(0.8, min(1.2, 1.0 - (ci_width / max_expected_width) * 0.2))
                
                # Ajustar confianza con factor bootstrap
                bootstrap_confidence = base_confidence * width_factor
                
                # Logging para debugging
                self.logger.debug(f"Legacy bootstrap confidence calculation - "
                                f"Base: {base_confidence:.2f}, "
                                f"CI: [{ci_lower:.2f}, {ci_upper:.2f}], "
                                f"Width: {ci_width:.2f}, "
                                f"Factor: {width_factor:.3f}, "
                                f"Final: {bootstrap_confidence:.2f}")
                
                return min(bootstrap_confidence, 100.0), bootstrap_info
            
        except ImportError:
            self.logger.debug("Bootstrap similarity module not available, using basic confidence")
        except Exception as e:
            self.logger.debug(f"Bootstrap confidence calculation failed: {e}")
        
        # Fallback a confianza básica
        return base_confidence, bootstrap_info
    
    def _calculate_geometric_consistency(self, kp1: List, kp2: List, 
                                       matches: List) -> float:
        """
        Calcula la consistencia geométrica de los matches
        
        Args:
            kp1: Keypoints de la primera imagen
            kp2: Keypoints de la segunda imagen
            matches: Lista de matches
            
        Returns:
            Consistencia geométrica (0-1)
        """
        try:
            if len(matches) < self.config.min_matches_for_ransac:
                return 0.0
            
            # Extraer puntos correspondientes
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Calcular homografía con RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                        self.config.ransac_threshold)
            
            if H is None or mask is None:
                return 0.0
            
            # Calcular proporción de inliers
            inliers = np.sum(mask)
            consistency = inliers / len(matches)
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculando consistencia geométrica: {e}")
            return 0.0
    
    def _calculate_match_quality(self, distances: List[float]) -> float:
        """
        Calcula la calidad de los matches basada en distancias
        
        Args:
            distances: Lista de distancias de matches
            
        Returns:
            Calidad de matches (0-1)
        """
        if not distances:
            return 0.0
            
        try:
            # Normalizar distancias (menor es mejor)
            max_dist = max(distances) if distances else 1.0
            if max_dist == 0:
                return 1.0
                
            # Invertir y normalizar (ahora mayor es mejor)
            normalized_distances = [1.0 - (d / max_dist) for d in distances]
            
            # Calcular calidad como promedio
            quality = np.mean(normalized_distances)
            
            return float(quality)
            
        except Exception as e:
            self.logger.error(f"Error calculando calidad de matches: {e}")
            return 0.0
    
    def compare_images(self, img1: np.ndarray, img2: np.ndarray,
                      algorithm: Optional[AlgorithmType] = None) -> MatchResult:
        """
        Compara dos imágenes
        
        Args:
            img1: Primera imagen
            img2: Segunda imagen
            algorithm: Algoritmo a utilizar (opcional)
            
        Returns:
            Resultado de la comparación
        """
        try:
            # Extraer características
            features1 = self.extract_features(img1, algorithm)
            features2 = self.extract_features(img2, algorithm)
            
            # Realizar matching
            result = self.match_features(features1, features2, algorithm)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error comparando imágenes: {e}")
            
            # Crear resultado de error
            return MatchResult(
                algorithm=algorithm.value if algorithm else self.config.algorithm.value,
                match_data={"error": str(e)}
            )
    
    def compare_image_files(self, img_path1: str, img_path2: str,
                          algorithm: Optional[AlgorithmType] = None) -> MatchResult:
        """
        Compara dos imágenes desde archivos
        
        Args:
            img_path1: Ruta de la primera imagen
            img_path2: Ruta de la segunda imagen
            algorithm: Algoritmo a utilizar (opcional)
            
        Returns:
            Resultado de la comparación
        """
        try:
            # Cargar imágenes
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            
            if img1 is None or img2 is None:
                self.logger.error(f"Error cargando imágenes: {img_path1} o {img_path2}")
                return MatchResult(
                    algorithm=algorithm.value if algorithm else self.config.algorithm.value,
                    match_data={"error": "Error cargando imágenes"}
                )
            
            # Convertir a RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Comparar imágenes
            result = self.compare_images(img1, img2, algorithm)
            
            # Añadir información de archivos
            result.match_data["file1"] = os.path.basename(img_path1)
            result.match_data["file2"] = os.path.basename(img_path2)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error comparando archivos de imagen: {e}")
            
            # Crear resultado de error
            return MatchResult(
                algorithm=algorithm.value if algorithm else self.config.algorithm.value,
                match_data={"error": str(e), "file1": os.path.basename(img_path1), 
                           "file2": os.path.basename(img_path2)}
            )
    
    def batch_compare(self, query_features: Dict[str, Any], 
                     database_features: List[Dict[str, Any]],
                     algorithm: Optional[AlgorithmType] = None, 
                     top_k: int = 5,
                     batch_size: int = 100,
                     memory_limit_mb: int = 500,
                     max_workers: int = None) -> List[MatchResult]:
        """
        Compara una imagen query contra múltiples imágenes de la base de datos
        con optimizaciones de memoria, procesamiento por lotes y paralelización
        
        Args:
            query_features: Características de la imagen query
            database_features: Lista de características de imágenes de la base de datos
            algorithm: Algoritmo a utilizar (opcional)
            top_k: Número de mejores resultados a retornar
            batch_size: Tamaño del lote para procesamiento
            memory_limit_mb: Límite de memoria en MB
            max_workers: Número máximo de hilos (None = auto)
            
        Returns:
            Lista de resultados de matching ordenados por similitud
        """
        import gc
        import psutil
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        import os
        
        # Determinar número óptimo de workers
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)  # Máximo 8 hilos
        
        results = []
        results_lock = Lock()
        process = psutil.Process()
        
        def process_single_comparison(args):
            """Función auxiliar para procesar una comparación individual"""
            db_features, global_index = args
            try:
                # Verificar que el algoritmo coincide
                if (algorithm is not None and 
                    db_features.get("algorithm") != algorithm.value):
                    return None
                    
                # Realizar matching
                match_result = self.match_features(query_features, db_features, algorithm)
                match_result.image1_id = 0  # Query
                match_result.image2_id = global_index  # DB index global
                
                return match_result
                
            except Exception as e:
                self.logger.error(f"Error comparando con imagen {global_index}: {e}")
                return None
        
        # Procesar en lotes para optimizar memoria
        for batch_start in range(0, len(database_features), batch_size):
            batch_end = min(batch_start + batch_size, len(database_features))
            batch_features = database_features[batch_start:batch_end]
            
            # Preparar argumentos para paralelización
            batch_args = [(db_features, batch_start + i) 
                         for i, db_features in enumerate(batch_features)]
            
            # Procesar lote en paralelo
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Enviar tareas
                future_to_index = {
                    executor.submit(process_single_comparison, args): args[1] 
                    for args in batch_args
                }
                
                # Recopilar resultados
                batch_results = []
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        if result is not None:
                            batch_results.append(result)
                    except Exception as e:
                        index = future_to_index[future]
                        self.logger.error(f"Error en hilo para imagen {index}: {e}")
            
            # Agregar resultados del lote de forma thread-safe
            with results_lock:
                results.extend(batch_results)
                
                # Mantener solo los top_k * 2 mejores para evitar acumulación excesiva
                if len(results) > top_k * 2:
                    results.sort(key=lambda x: x.similarity_score, reverse=True)
                    results = results[:top_k * 2]
            
            # Verificar uso de memoria después de cada lote
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > memory_limit_mb:
                self.logger.warning(f"Límite de memoria alcanzado: {memory_mb:.1f}MB")
                gc.collect()  # Forzar garbage collection
            
            # Limpiar memoria después de cada lote
            del batch_results, batch_args
            gc.collect()
        
        # Ordenar por score de similitud y retornar top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def calculate_cmc_curve(self, query_results: List[List[MatchResult]], 
                          ground_truth: List[int]) -> Dict[str, List[float]]:
        """
        Calcula curva CMC (Cumulative Match Characteristic)
        
        Args:
            query_results: Lista de resultados de query (cada elemento es una lista de MatchResult)
            ground_truth: Lista de índices de ground truth
            
        Returns:
            Diccionario con curva CMC
        """
        if len(query_results) != len(ground_truth):
            raise ValueError("Número de queries no coincide con ground truth")
        
        max_rank = min(len(results) for results in query_results) if query_results else 0
        cmc_scores = []
        
        for rank in range(1, max_rank + 1):
            correct_at_rank = 0
            
            for query_idx, (results, true_match) in enumerate(zip(query_results, ground_truth)):
                # Verificar si el match correcto está en el top-rank
                top_rank_ids = [r.image2_id for r in results[:rank]]
                if true_match in top_rank_ids:
                    correct_at_rank += 1
            
            accuracy_at_rank = correct_at_rank / len(query_results)
            cmc_scores.append(accuracy_at_rank)
        
        return {
            "ranks": list(range(1, max_rank + 1)),
            "accuracies": cmc_scores
        }
    
    def create_match_visualization(self, img1: np.ndarray, img2: np.ndarray,
                                 kp1: List, kp2: List, matches: List,
                                 max_matches: Optional[int] = None) -> np.ndarray:
        """
        Crea visualización de matches entre dos imágenes
        
        Args:
            img1: Primera imagen
            img2: Segunda imagen
            kp1: Keypoints de la primera imagen
            kp2: Keypoints de la segunda imagen
            matches: Lista de matches
            max_matches: Número máximo de matches a mostrar
            
        Returns:
            Imagen con visualización de matches
        """
        # Validar entradas
        if img1 is None or img2 is None:
            raise ValueError("Las imágenes no pueden ser None")
            
        if max_matches is None:
            max_matches = self.config.max_matches_to_draw
            
        # Verificar si hay keypoints y matches válidos
        if not kp1 or not kp2 or not matches:
            # Crear imagen combinada sin matches
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Crear imagen combinada
            combined_height = max(h1, h2)
            combined_width = w1 + w2
            
            if len(img1.shape) == 3:
                combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
                combined_img[:h1, :w1] = img1
                combined_img[:h2, w1:w1+w2] = img2
            else:
                combined_img = np.zeros((combined_height, combined_width), dtype=np.uint8)
                combined_img[:h1, :w1] = img1
                combined_img[:h2, w1:w1+w2] = img2
                # Convertir a BGR para poder agregar texto en color
                combined_img = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
            
            # Agregar texto informativo
            cv2.putText(combined_img, "No matches found", 
                       (combined_width//4, combined_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return combined_img
            
        # Limitar número de matches para visualización
        matches_to_show = matches[:max_matches] if len(matches) > max_matches else matches
        
        # Crear imagen de matches usando OpenCV
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches_to_show,
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_img
    
    def create_keypoints_visualization(self, img: np.ndarray, keypoints: List) -> np.ndarray:
        """
        Crea visualización de keypoints en una imagen
        
        Args:
            img: Imagen
            keypoints: Lista de keypoints
            
        Returns:
            Imagen con visualización de keypoints
        """
        if img is None:
            raise ValueError("La imagen no puede ser None")
            
        vis_img = img.copy()
        if len(vis_img.shape) == 3:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Solo dibujar keypoints si existen
        if keypoints and len(keypoints) > 0:
            # Dibujar keypoints
            vis_img = cv2.drawKeypoints(
                vis_img, keypoints, None, 
                color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        else:
            # Si no hay keypoints, agregar texto informativo
            height, width = vis_img.shape[:2]
            cv2.putText(vis_img, "No keypoints detected", 
                       (width//4, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        return vis_img
    
    def export_match_report(self, match_result: MatchResult, 
                          output_path: str) -> bool:
        """
        Exporta reporte de matching a archivo JSON
        
        Args:
            match_result: Resultado de matching
            output_path: Ruta de salida
            
        Returns:
            True si se exportó correctamente, False en caso contrario
        """
        try:
            # Convertir dataclass a diccionario
            report_data = {
                "match_result": {
                    "algorithm": match_result.algorithm,
                    "similarity_score": match_result.similarity_score,
                    "confidence": match_result.confidence,
                    "total_keypoints1": match_result.total_keypoints1,
                    "total_keypoints2": match_result.total_keypoints2,
                    "total_matches": match_result.total_matches,
                    "good_matches": match_result.good_matches,
                    "processing_time": match_result.processing_time,
                    "geometric_consistency": match_result.geometric_consistency,
                    "match_quality": match_result.match_quality,
                    "keypoint_density": match_result.keypoint_density
                },
                "match_data": match_result.match_data,
                "descriptor_distance_stats": match_result.descriptor_distance_stats,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Guardar reporte
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Reporte de matching exportado: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exportando reporte: {e}")
            return False
    
    def evaluate_algorithms(self, image_pairs: List[Tuple[np.ndarray, np.ndarray, bool]]) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de diferentes algoritmos
        
        Args:
            image_pairs: Lista de pares de imágenes con etiqueta (img1, img2, same_weapon)
            
        Returns:
            Diccionario con resultados de evaluación
        """
        results = {}
        
        # Evaluar cada algoritmo disponible
        for alg in AlgorithmType:
            if alg not in self.feature_detectors:
                continue
                
            self.logger.info(f"Evaluando algoritmo: {alg.value}")
            
            # Resultados para este algoritmo
            alg_results = {
                "same_weapon": [],
                "different_weapon": [],
                "processing_times": [],
                "keypoint_counts": [],
                "match_counts": [],
                "quality_scores": []
            }
            
            # Procesar cada par de imágenes
            for img1, img2, same_weapon in image_pairs:
                # Comparar imágenes
                result = self.compare_images(img1, img2, alg)
                
                # Guardar resultados
                if same_weapon:
                    alg_results["same_weapon"].append(result.similarity_score)
                else:
                    alg_results["different_weapon"].append(result.similarity_score)
                    
                alg_results["processing_times"].append(result.processing_time)
                alg_results["keypoint_counts"].append((result.total_keypoints1 + result.total_keypoints2) / 2)
                alg_results["match_counts"].append(result.good_matches)
                alg_results["quality_scores"].append(result.match_quality)
            
            # Calcular estadísticas
            results[alg.value] = {
                "mean_similarity_same": np.mean(alg_results["same_weapon"]) if alg_results["same_weapon"] else 0,
                "std_similarity_same": np.std(alg_results["same_weapon"]) if alg_results["same_weapon"] else 0,
                "mean_similarity_different": np.mean(alg_results["different_weapon"]) if alg_results["different_weapon"] else 0,
                "std_similarity_different": np.std(alg_results["different_weapon"]) if alg_results["different_weapon"] else 0,
                "discrimination_score": (np.mean(alg_results["same_weapon"]) - np.mean(alg_results["different_weapon"])) 
                                      if alg_results["same_weapon"] and alg_results["different_weapon"] else 0,
                "mean_processing_time": np.mean(alg_results["processing_times"]),
                "mean_keypoints": np.mean(alg_results["keypoint_counts"]),
                "mean_matches": np.mean(alg_results["match_counts"]),
                "mean_quality": np.mean(alg_results["quality_scores"]) if alg_results["quality_scores"] else 0
            }
        
        # Calcular resumen
        summary = {
            "best_similarity": max(results.items(), key=lambda x: x[1]["mean_similarity_same"])[0] 
                             if results else None,
            "best_discrimination": max(results.items(), key=lambda x: x[1]["discrimination_score"])[0] 
                                if results else None,
            "fastest": min(results.items(), key=lambda x: x[1]["mean_processing_time"])[0] 
                     if results else None
        }
        
        return {
            "algorithms": results,
            "summary": summary,
            "total_image_pairs": len(image_pairs),
            "same_weapon_pairs": sum(1 for _, _, same in image_pairs if same),
            "different_weapon_pairs": sum(1 for _, _, same in image_pairs if not same)
        }

# Función de ayuda para crear configuración desde string
def create_matching_config(level: str = "standard") -> MatchingConfig:
    """
    Crea una configuración de matching a partir de un nivel
    
    Args:
        level: Nivel de matching (basic, standard, advanced)
        
    Returns:
        Configuración de matching
    """
    level_enum = next((l for l in MatchingLevel if l.value == level.lower()), 
                     MatchingLevel.STANDARD)
    
    matcher = UnifiedMatcher()
    return matcher.default_configs[level_enum]

# Punto de entrada para uso como script
if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description="Matcher unificado para imágenes balísticas")
    parser.add_argument("--mode", choices=["compare", "evaluate"], default="compare",
                      help="Modo de operación")
    
    # Argumentos para modo compare
    parser.add_argument("--img1", help="Primera imagen para comparación")
    parser.add_argument("--img2", help="Segunda imagen para comparación")
    parser.add_argument("--output", "-o", help="Archivo de salida para reporte")
    parser.add_argument("--level", "-l", help="Nivel de matching", 
                       choices=["basic", "standard", "advanced"], default="standard")
    parser.add_argument("--algorithm", "-a", help="Algoritmo a utilizar",
                       choices=["ORB", "SIFT", "AKAZE", "BRISK", "KAZE", "CMC"], default="ORB")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Guardar visualización de matches")
    
    # Argumentos para modo evaluate
    parser.add_argument("--dir", help="Directorio con imágenes para evaluación")
    
    args = parser.parse_args()
    
    # Crear matcher
    config = create_matching_config(args.level)
    config.algorithm = next((a for a in AlgorithmType if a.value == args.algorithm), 
                           AlgorithmType.ORB)
    matcher = UnifiedMatcher(config)
    
    if args.mode == "compare":
        # Verificar argumentos
        if not args.img1 or not args.img2:
            print("Error: Debe especificar dos imágenes para comparar")
            exit(1)
        
        # Comparar imágenes
        result = matcher.compare_image_files(args.img1, args.img2)
        
        # Mostrar resultados
        print("\n=== RESULTADOS DE COMPARACIÓN ===")
        print(f"Algoritmo: {result.algorithm}")
        print(f"Score de similitud: {result.similarity_score:.2f}")
        print(f"Confianza: {result.confidence:.2f}")
        print(f"Keypoints: {result.total_keypoints1} / {result.total_keypoints2}")
        print(f"Matches: {result.good_matches} / {result.total_matches}")
        print(f"Consistencia geométrica: {result.geometric_consistency:.3f}")
        print(f"Calidad de matches: {result.match_quality:.3f}")
        print(f"Tiempo de procesamiento: {result.processing_time:.3f}s")
        
        # Guardar reporte si se especifica
        if args.output:
            matcher.export_match_report(result, args.output)
            print(f"Reporte guardado en: {args.output}")
        
        # Guardar visualización si se solicita
        if args.visualize:
            # Cargar imágenes
            img1 = cv2.imread(args.img1)
            img2 = cv2.imread(args.img2)
            
            if img1 is not None and img2 is not None:
                # Extraer características
                features1 = matcher.extract_features(img1)
                features2 = matcher.extract_features(img2)
                
                # Obtener matches
                matches = []
                for match_data in result.match_data.get("matches", []):
                    m = cv2.DMatch()
                    m.queryIdx = match_data["kp1_idx"]
                    m.trainIdx = match_data["kp2_idx"]
                    m.distance = match_data["distance"]
                    matches.append(m)
                
                # Crear visualización
                vis_img = matcher.create_match_visualization(
                    img1, img2, features1["keypoints"], features2["keypoints"], matches
                )
                
                # Guardar visualización
                vis_path = f"{os.path.splitext(args.output)[0]}_vis.png" if args.output else "matches_vis.png"
                cv2.imwrite(vis_path, vis_img)
                print(f"Visualización guardada en: {vis_path}")
    
    elif args.mode == "evaluate":
        # Verificar argumentos
        if not args.dir:
            print("Error: Debe especificar un directorio con imágenes para evaluación")
            exit(1)
        
        # TODO: Implementar carga de imágenes y evaluación
        print("Modo de evaluación no implementado completamente")
        
    else:
        print(f"Modo no válido: {args.mode}")
        exit(1)