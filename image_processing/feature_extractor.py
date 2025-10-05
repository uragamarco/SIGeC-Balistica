"""
Feature Extractor API - Módulo especializado como servidor API
Delega el procesamiento principal al motor optimizado ballistic_features.py
Mantiene compatibilidad con endpoints existentes
"""

import cv2
import numpy as np
import logging
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from flask import Flask, request, jsonify
from scipy import ndimage
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy

# Importar el motor de procesamiento balístico optimizado
try:
    from ballistic_features import BallisticFeatureExtractor, BallisticFeatures, ParallelConfig
    BALLISTIC_ENGINE_AVAILABLE = True
except ImportError:
    BALLISTIC_ENGINE_AVAILABLE = False
    print("Warning: Motor de características balísticas no disponible")

# Importar cache LBP si está disponible
try:
    from lbp_cache import cached_local_binary_pattern, get_lbp_cache_stats
    LBP_CACHE_AVAILABLE = True
except ImportError:
    LBP_CACHE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# CONFIGURACIÓN Y UTILIDADES
# ============================================================================

def convert_numpy_types(obj):
    """Convierte recursivamente tipos NumPy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FeatureExtractorAPI')

# ============================================================================
# EXCEPCIONES PERSONALIZADAS
# ============================================================================

class FeatureExtractionError(Exception):
    """Excepción base para errores de extracción de características"""
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message)
        self.details = details or {}

class ImageValidationError(FeatureExtractionError):
    """Error de validación de imagen"""
    pass

class ProcessingError(FeatureExtractionError):
    """Error durante el procesamiento"""
    pass

class FeatureExtractionTimeout(FeatureExtractionError):
    """Timeout durante la extracción"""
    pass

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_image(image: np.ndarray) -> None:
    """Valida que la imagen sea válida para procesamiento"""
    if image is None:
        raise ImageValidationError("La imagen es None")
    
    if not isinstance(image, np.ndarray):
        raise ImageValidationError("La imagen debe ser un numpy array")
    
    if image.size == 0:
        raise ImageValidationError("La imagen está vacía")
    
    if len(image.shape) < 2:
        raise ImageValidationError("La imagen debe tener al menos 2 dimensiones")
    
    if image.shape[0] < 10 or image.shape[1] < 10:
        raise ImageValidationError("La imagen es demasiado pequeña (mínimo 10x10)")
    
    if image.shape[0] > 10000 or image.shape[1] > 10000:
        raise ImageValidationError("La imagen es demasiado grande (máximo 10000x10000)")

# ============================================================================
# FUNCIONES PRINCIPALES DE EXTRACCIÓN (delegadas al motor optimizado)
# ============================================================================

def calculate_ballistic_features(image_data: bytes) -> dict:
    """Función principal para cálculo de características balísticas - usa motor optimizado"""
    try:
        # Decodificar imagen
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ProcessingError("No se pudo decodificar la imagen")
        
        validate_image(image)
        
        if BALLISTIC_ENGINE_AVAILABLE:
            # Usar motor optimizado
            extractor = BallisticFeatureExtractor()
            
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image
            
            # Extraer todas las características usando el motor optimizado
            all_features = extractor.extract_all_features(gray_img, 'cartridge_case')
            
            # Formatear para compatibilidad con API existente
            return {
                'ballistic_features': all_features.get('ballistic_features', {}),
                'quality_metrics': all_features.get('quality_metrics', {}),
                'processing_info': {
                    'engine': 'optimized_ballistic_engine',
                    'version': '2.0',
                    'parallel_processing': True
                }
            }
        else:
            # Fallback a implementación básica
            return _calculate_ballistic_features_basic(image)
            
    except Exception as e:
        logger.error(f"Error calculando características balísticas: {e}")
        raise ProcessingError(f"Error en cálculo balístico: {str(e)}")

def extract_features(image_path: str) -> dict:
    """Función de compatibilidad para extracción de características desde archivo"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        
        validate_image(image)
        
        if BALLISTIC_ENGINE_AVAILABLE:
            extractor = BallisticFeatureExtractor()
            
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image
            
            # Extraer características usando motor optimizado
            all_features = extractor.extract_all_features(gray_img, 'cartridge_case')
            ballistic_features = all_features.get('ballistic_features', {})
            
            # Formatear para compatibilidad con formato legacy
            return {
                'hu_moments': ballistic_features.get('hu_moments', []),
                'contour_area': ballistic_features.get('contour_area', 0.0),
                'contour_len': ballistic_features.get('contour_len', 0.0),
                'lbp_uniformity': ballistic_features.get('lbp_uniformity', 0.0),
                'firing_pin_marks': {
                    'num_marks': ballistic_features.get('firing_pin', {}).get('num_marks', 0),
                    'avg_mark_radius': ballistic_features.get('firing_pin', {}).get('avg_radius', 0.0),
                    'mark_positions': ballistic_features.get('firing_pin', {}).get('positions', []),
                    'mark_intensities': ballistic_features.get('firing_pin', {}).get('intensities', [])
                },
                'striation_patterns': {
                    'num_lines': ballistic_features.get('striation', {}).get('num_lines', 0),
                    'dominant_directions': ballistic_features.get('striation', {}).get('dominant_directions', []),
                    'patterns': []  # Para compatibilidad
                }
            }
        else:
            # Fallback básico
            return _extract_features_basic(image)
            
    except Exception as e:
        logger.error(f"Error extrayendo características: {e}")
        raise ProcessingError(f"Error en extracción: {str(e)}")

# ============================================================================
# FUNCIONES DE COMPATIBILIDAD ESPECÍFICAS
# ============================================================================

def extract_sift_features(gray_img, max_features=500, contrast_threshold=0.04):
    """Extrae características SIFT - mantenido para compatibilidad"""
    try:
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        
        if descriptors is not None:
            return {
                'keypoints': len(keypoints),
                'descriptors_shape': descriptors.shape,
                'descriptor_stats': {
                    'mean': float(np.mean(descriptors)),
                    'std': float(np.std(descriptors)),
                    'min': float(np.min(descriptors)),
                    'max': float(np.max(descriptors))
                }
            }
        else:
            return {'keypoints': 0, 'descriptors_shape': (0, 0), 'descriptor_stats': {}}
    except Exception as e:
        logger.warning(f"Error en SIFT: {e}")
        return {'keypoints': 0, 'descriptors_shape': (0, 0), 'descriptor_stats': {}}

def extract_orb_features(gray_img, max_features=500):
    """Extrae características ORB - mantenido para compatibilidad"""
    try:
        orb = cv2.ORB_create(nfeatures=max_features)
        keypoints, descriptors = orb.detectAndCompute(gray_img, None)
        
        if descriptors is not None:
            return {
                'keypoints': len(keypoints),
                'descriptors_shape': descriptors.shape,
                'descriptor_stats': {
                    'mean': float(np.mean(descriptors)),
                    'std': float(np.std(descriptors))
                }
            }
        else:
            return {'keypoints': 0, 'descriptors_shape': (0, 0), 'descriptor_stats': {}}
    except Exception as e:
        logger.warning(f"Error en ORB: {e}")
        return {'keypoints': 0, 'descriptors_shape': (0, 0), 'descriptor_stats': {}}

def extract_advanced_lbp(gray_img, radius=3, n_points=24):
    """Extrae características LBP avanzadas"""
    try:
        if LBP_CACHE_AVAILABLE:
            lbp = cached_local_binary_pattern(gray_img, n_points, radius, method='uniform')
        else:
            lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        return {
            'histogram': hist.tolist(),
            'uniformity': float(np.sum(hist[:-1])),
            'entropy': float(shannon_entropy(hist))
        }
    except Exception as e:
        logger.warning(f"Error en LBP: {e}")
        return {'histogram': [], 'uniformity': 0.0, 'entropy': 0.0}

# ============================================================================
# ENDPOINTS DE API FLASK
# ============================================================================

@app.route('/extract_ballistic', methods=['POST'])
def extract_ballistic_endpoint():
    """Endpoint especializado para extracción de características balísticas"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        # Leer imagen
        image_data = file.read()
        features = calculate_ballistic_features(image_data)
        
        # Convertir tipos NumPy para JSON
        features_json = convert_numpy_types(features)
        
        return jsonify({
            'success': True,
            'features': features_json,
            'filename': file.filename,
            'processing_time': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint balístico: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract_advanced_ballistic', methods=['POST'])
def extract_advanced_ballistic_endpoint():
    """Endpoint avanzado con configuración de paralelización"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        # Parámetros de configuración
        use_parallel = request.form.get('use_parallel', 'true').lower() == 'true'
        specimen_type = request.form.get('specimen_type', 'cartridge_case')
        max_workers = int(request.form.get('max_workers', 2))
        
        if not BALLISTIC_ENGINE_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Motor de características balísticas no disponible'
            }), 503
        
        # Decodificar imagen
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        # Configurar paralelización
        parallel_config = ParallelConfig(
            max_workers_process=max_workers,
            max_workers_thread=max_workers,
            enable_gabor_parallel=True,
            enable_roi_parallel=True
        )
        
        # Extraer características
        extractor = BallisticFeatureExtractor(parallel_config)
        
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        
        start_time = time.time()
        ballistic_features = extractor.extract_ballistic_features(
            gray_img, specimen_type, use_parallel
        )
        processing_time = time.time() - start_time
        
        # Convertir a diccionario para JSON
        features_dict = {
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
                'frequency': ballistic_features.striation_frequency,
                'num_lines': ballistic_features.striation_num_lines,
                'dominant_directions': ballistic_features.striation_dominant_directions,
                'parallelism_score': ballistic_features.striation_parallelism_score
            },
            'quality': {
                'score': ballistic_features.quality_score,
                'confidence': ballistic_features.confidence
            }
        }
        
        return jsonify({
            'success': True,
            'ballistic_features': convert_numpy_types(features_dict),
            'processing_info': {
                'processing_time': processing_time,
                'parallel_processing': use_parallel,
                'specimen_type': specimen_type,
                'engine_version': '2.0_hybrid'
            },
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint balístico avanzado: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract', methods=['POST'])
def extract_features_endpoint():
    """Endpoint de compatibilidad para extracción general"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        # Usar función de compatibilidad
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        # Simular extracción usando motor optimizado
        features = calculate_ballistic_features(image_data)
        
        return jsonify({
            'success': True,
            'features': convert_numpy_types(features),
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint general: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract_sift', methods=['POST'])
def extract_sift_endpoint():
    """Endpoint para características SIFT"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        features = extract_sift_features(image)
        
        return jsonify({
            'success': True,
            'sift_features': convert_numpy_types(features),
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint SIFT: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract_orb', methods=['POST'])
def extract_orb_endpoint():
    """Endpoint para características ORB"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        features = extract_orb_features(image)
        
        return jsonify({
            'success': True,
            'orb_features': convert_numpy_types(features),
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint ORB: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/extract_lbp', methods=['POST'])
def extract_lbp_endpoint():
    """Endpoint para características LBP"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400
        
        features = extract_advanced_lbp(image)
        
        return jsonify({
            'success': True,
            'lbp_features': convert_numpy_types(features),
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error en endpoint LBP: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Endpoint de verificación de salud del API"""
    return jsonify({
        'status': 'healthy',
        'ballistic_engine': BALLISTIC_ENGINE_AVAILABLE,
        'lbp_cache': LBP_CACHE_AVAILABLE,
        'timestamp': time.time(),
        'version': '2.0.0'
    })

@app.route('/api/extract', methods=['POST'])
def api_extract_features():
    """Endpoint principal del API para extracción de características"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
        # Leer imagen
        image_data = file.read()
        features = calculate_ballistic_features(image_data)
        
        return jsonify({
            'success': True,
            'features': convert_numpy_types(features),
            'filename': file.filename,
            'processing_time': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error en API extract: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# FUNCIONES DE FALLBACK BÁSICAS
# ============================================================================

def _calculate_ballistic_features_basic(image):
    """Implementación básica para fallback"""
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image
    
    # Implementación muy básica
    return {
        'ballistic_features': {
            'firing_pin': {'num_marks': 0, 'avg_radius': 0.0},
            'striation': {'num_lines': 0, 'density': 0.0},
            'breech_face': {'roughness': 0.0, 'orientation': 0.0}
        },
        'quality_metrics': {
            'quality_score': 0.5,
            'confidence': 0.5
        },
        'processing_info': {
            'engine': 'basic_fallback',
            'version': '1.0',
            'parallel_processing': False
        }
    }

def _extract_features_basic(image):
    """Extracción básica para compatibilidad"""
    return {
        'hu_moments': [0.0] * 7,
        'contour_area': 0.0,
        'contour_len': 0.0,
        'lbp_uniformity': 0.0,
        'firing_pin_marks': {
            'num_marks': 0,
            'avg_mark_radius': 0.0,
            'mark_positions': [],
            'mark_intensities': []
        },
        'striation_patterns': {
            'num_lines': 0,
            'dominant_directions': [],
            'patterns': []
        }
    }

# ============================================================================
# CLASE WRAPPER PARA COMPATIBILIDAD
# ============================================================================

class FeatureExtractor:
    """Clase wrapper para compatibilidad con código existente"""
    
    def __init__(self):
        """Inicializar el extractor de características"""
        self.ballistic_available = BALLISTIC_ENGINE_AVAILABLE
        self.lbp_cache_available = LBP_CACHE_AVAILABLE
        
    def extract_features(self, image_path: str) -> dict:
        """Extraer características de una imagen"""
        return extract_features(image_path)
    
    def extract_features_from_array(self, image_array: np.ndarray) -> dict:
        """Extraer características de un array de imagen"""
        if BALLISTIC_ENGINE_AVAILABLE:
            try:
                extractor = BallisticFeatureExtractor()
                
                if len(image_array.shape) == 3:
                    gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                else:
                    gray_img = image_array
                
                # Extraer características usando motor optimizado
                all_features = extractor.extract_all_features(gray_img, 'cartridge_case')
                ballistic_features = all_features.get('ballistic_features', {})
                
                # Formatear para compatibilidad con formato legacy
                return {
                    'hu_moments': ballistic_features.get('hu_moments', []),
                    'contour_area': ballistic_features.get('contour_area', 0.0),
                    'contour_len': ballistic_features.get('contour_len', 0.0),
                    'lbp_uniformity': ballistic_features.get('lbp_uniformity', 0.0),
                    'firing_pin_marks': {
                        'num_marks': ballistic_features.get('firing_pin', {}).get('num_marks', 0),
                        'avg_mark_radius': ballistic_features.get('firing_pin', {}).get('avg_radius', 0.0),
                        'mark_positions': ballistic_features.get('firing_pin', {}).get('positions', []),
                        'mark_intensities': ballistic_features.get('firing_pin', {}).get('intensities', [])
                    },
                    'striation_patterns': {
                        'num_lines': ballistic_features.get('striation', {}).get('num_lines', 0),
                        'dominant_directions': ballistic_features.get('striation', {}).get('dominant_directions', []),
                        'patterns': []  # Para compatibilidad
                    }
                }
            except Exception as e:
                logger.error(f"Error extrayendo características con motor balístico: {e}")
                return _extract_features_basic(image_array)
        else:
            # Fallback básico
            return _extract_features_basic(image_array)
    
    def extract_sift_features(self, image_array: np.ndarray, max_features=500, contrast_threshold=0.04) -> dict:
        """Extraer características SIFT"""
        if len(image_array.shape) == 3:
            gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image_array
        return extract_sift_features(gray_img, max_features, contrast_threshold)
    
    def extract_orb_features(self, image_array: np.ndarray, max_features=500) -> dict:
        """Extraer características ORB"""
        if len(image_array.shape) == 3:
            gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image_array
        return extract_orb_features(gray_img, max_features)
    
    def extract_lbp_features(self, image_array: np.ndarray, radius=3, n_points=24) -> dict:
        """Extraer características LBP avanzadas"""
        if len(image_array.shape) == 3:
            gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image_array
        return extract_advanced_lbp(gray_img, radius, n_points)
    
    def is_available(self) -> bool:
        """Verificar si el extractor está disponible"""
        return True  # Siempre disponible con fallback básico
    
    def get_status(self) -> dict:
        """Obtener estado del extractor"""
        return {
            'available': True,
            'ballistic_engine': self.ballistic_available,
            'lbp_cache': self.lbp_cache_available,
            'version': '1.0'
        }

# ============================================================================
# MAIN Y CONFIGURACIÓN
# ============================================================================

if __name__ == '__main__':
    logger.info("Iniciando Feature Extractor API Server (Especializado)")
    logger.info(f"Motor balístico optimizado: {'Disponible' if BALLISTIC_ENGINE_AVAILABLE else 'No disponible'}")
    logger.info(f"Cache LBP: {'Disponible' if LBP_CACHE_AVAILABLE else 'No disponible'}")
    
    if len(sys.argv) > 1:
        # Modo línea de comandos (mantenido para compatibilidad)
        parser = argparse.ArgumentParser(description='API de extracción de características balísticas')
        parser.add_argument('image_path', type=str, help='Ruta a la imagen para analizar')
        parser.add_argument('--specimen-type', default='cartridge_case', help='Tipo de espécimen')
        parser.add_argument('--use-parallel', action='store_true', help='Usar procesamiento paralelo')
        args = parser.parse_args()
        
        try:
            # Procesar imagen desde línea de comandos
            features = extract_features(args.image_path)
            
            # Formatear salida para compatibilidad
            firing_pin_marks_formatted = []
            if "firing_pin_marks" in features and features["firing_pin_marks"]["mark_positions"]:
                positions = features["firing_pin_marks"]["mark_positions"]
                avg_radius = features["firing_pin_marks"].get("avg_mark_radius", 5.0)
                
                for pos in positions:
                    if len(pos) >= 2:
                        firing_pin_marks_formatted.append({
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "radius": float(avg_radius)
                        })
            
            striation_patterns_formatted = []
            if "striation_patterns" in features and features["striation_patterns"]["patterns"]:
                patterns = features["striation_patterns"]["patterns"]
                for pattern in patterns:
                    if len(pattern) >= 3:
                        striation_patterns_formatted.append({
                            "angle": float(pattern[0]),
                            "length": float(pattern[1]),
                            "strength": float(pattern[2])
                        })
            
            response = {
                "hu_moments": features["hu_moments"],
                "contour_area": features["contour_area"],
                "contour_len": features["contour_len"],
                "lbp_uniformity": features["lbp_uniformity"],
                "firing_pin_marks": firing_pin_marks_formatted,
                "striation_patterns": striation_patterns_formatted,
                "filename": os.path.basename(args.image_path),
                "content_type": "image/tiff" if args.image_path.lower().endswith(('.tif', '.tiff')) else "image/jpeg",
                "file_size": os.path.getsize(args.image_path)
            }
            
            print(json.dumps(convert_numpy_types(response), indent=2))
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            print(json.dumps({"error": str(e)}, indent=2))
            sys.exit(1)
    else:
        # Modo servidor
        logger.info("Iniciando servidor Flask en puerto 5000...")
        try:
            app.run(host='0.0.0.0', port=5000, debug=False)
        except Exception as e:
            logger.error(f"Error iniciando servidor Flask: {e}")
            sys.exit(1)
    