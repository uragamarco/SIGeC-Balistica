#!/usr/bin/env python3
"""
SIGeC-Balistica- Implementaciones de Fallback Robustas
===============================================

Este módulo contiene implementaciones de fallback para dependencias opcionales,
proporcionando funcionalidad alternativa cuando las librerías principales no están disponibles.

Autor: SIGeC-BalisticaTeam
Versión: 1.0.0
"""

import numpy as np
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DeepLearningFallback:
    """
    Fallback para funcionalidades de Deep Learning (PyTorch/TensorFlow)
    Proporciona funcionalidad básica usando NumPy y scikit-learn
    """
    
    def __init__(self):
        self.available = False
        self.backend = "numpy"
        logger.warning("Deep Learning libraries no disponibles - usando fallback con NumPy")
    
    def create_model(self, input_shape: Tuple[int, ...], num_classes: int = 2):
        """Crea un modelo simple usando scikit-learn"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            
            # Usar MLP como aproximación a una red neuronal
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            logger.info("Usando MLPClassifier como fallback para deep learning")
            return model
            
        except ImportError:
            logger.warning("scikit-learn no disponible - funcionalidad de ML muy limitada")
            return None
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocesa características para el modelo fallback"""
        # Normalización simple
        if features.ndim > 2:
            # Aplanar características multidimensionales
            features = features.reshape(features.shape[0], -1)
        
        # Normalización min-max
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        
        # Evitar división por cero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (features - min_vals) / range_vals
        return normalized
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extrae características básicas de una imagen"""
        try:
            from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
            from skimage.measure import shannon_entropy
            
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                image = np.mean(image, axis=2)
            
            features = []
            
            # Características estadísticas básicas
            features.extend([
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image),
                shannon_entropy(image)
            ])
            
            # Local Binary Pattern
            lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
            features.extend(lbp_hist / np.sum(lbp_hist))  # Normalizar
            
            # Matriz de co-ocurrencia (GLCM)
            image_int = (image * 255).astype(np.uint8)
            glcm = graycomatrix(image_int, [1], [0], levels=256, symmetric=True, normed=True)
            
            # Propiedades GLCM
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0]
            ])
            
            return np.array(features)
            
        except ImportError as e:
            logger.warning(f"Algunas características no disponibles: {e}")
            # Fallback básico con solo estadísticas
            return np.array([
                np.mean(image),
                np.std(image),
                np.min(image),
                np.max(image)
            ])


class WebServiceFallback:
    """
    Fallback para servicios web (Flask/FastAPI)
    Proporciona funcionalidad básica de servidor usando http.server
    """
    
    def __init__(self):
        self.available = False
        self.server = None
        logger.warning("Flask no disponible - usando servidor HTTP básico")
    
    def create_simple_server(self, port: int = 8000):
        """Crea un servidor HTTP simple"""
        import http.server
        import socketserver
        from threading import Thread
        
        class SimpleHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok", "service": "SIGeC-Balistica"}')
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Servicio web limitado - Flask no disponible')
        
        try:
            with socketserver.TCPServer(("", port), SimpleHandler) as httpd:
                logger.info(f"Servidor HTTP básico iniciado en puerto {port}")
                self.server = httpd
                return httpd
        except Exception as e:
            logger.error(f"Error iniciando servidor: {e}")
            return None
    
    def stop_server(self):
        """Detiene el servidor"""
        if self.server:
            self.server.shutdown()
            logger.info("Servidor HTTP detenido")


class ImageProcessingFallback:
    """
    Fallback para procesamiento avanzado de imágenes
    """
    
    def __init__(self):
        self.available = False
        logger.warning("Algunas librerías de procesamiento de imágenes no disponibles")
    
    def read_raw_image(self, filepath: str) -> Optional[np.ndarray]:
        """Fallback para lectura de imágenes RAW"""
        try:
            # Intentar con PIL/Pillow primero
            from PIL import Image
            img = Image.open(filepath)
            return np.array(img)
        except Exception as e:
            logger.warning(f"No se pudo leer imagen RAW {filepath}: {e}")
            return None
    
    def advanced_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Reducción de ruido básica sin librerías avanzadas"""
        try:
            from scipy import ndimage
            # Filtro gaussiano simple
            return ndimage.gaussian_filter(image, sigma=1.0)
        except ImportError:
            # Fallback con convolución manual
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            if len(image.shape) == 2:
                return self._convolve2d(image, kernel)
            else:
                # Aplicar a cada canal
                result = np.zeros_like(image)
                for i in range(image.shape[2]):
                    result[:, :, i] = self._convolve2d(image[:, :, i], kernel)
                return result
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolución 2D básica"""
        h, w = image.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Padding
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result


class DatabaseFallback:
    """
    Fallback para funcionalidades avanzadas de base de datos
    """
    
    def __init__(self):
        self.available = False
        self.vector_storage = {}
        logger.warning("FAISS no disponible - usando almacenamiento vectorial básico")
    
    def create_vector_index(self, dimension: int):
        """Crea un índice vectorial simple"""
        return SimpleVectorIndex(dimension)
    
    def similarity_search(self, query_vector: np.ndarray, vectors: List[np.ndarray], k: int = 5) -> List[Tuple[int, float]]:
        """Búsqueda de similitud básica usando distancia euclidiana"""
        distances = []
        
        for i, vector in enumerate(vectors):
            distance = np.linalg.norm(query_vector - vector)
            distances.append((i, distance))
        
        # Ordenar por distancia (menor = más similar)
        distances.sort(key=lambda x: x[1])
        return distances[:k]


class SimpleVectorIndex:
    """Índice vectorial simple como fallback para FAISS"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ids = []
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """Añade vectores al índice"""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        for i, vector in enumerate(vectors):
            self.vectors.append(vector)
            if ids:
                self.ids.append(ids[i])
            else:
                self.ids.append(len(self.vectors) - 1)
    
    def search(self, query_vectors: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Busca vectores similares"""
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        distances = []
        indices = []
        
        for query in query_vectors:
            query_distances = []
            for i, vector in enumerate(self.vectors):
                dist = np.linalg.norm(query - vector)
                query_distances.append((dist, i))
            
            # Ordenar y tomar los k mejores
            query_distances.sort(key=lambda x: x[0])
            query_distances = query_distances[:k]
            
            distances.append([d[0] for d in query_distances])
            indices.append([d[1] for d in query_distances])
        
        return np.array(distances), np.array(indices)


class FallbackRegistry:
    """
    Registro centralizado de implementaciones de fallback
    """
    
    def __init__(self):
        self.fallbacks = {
            'deep_learning': DeepLearningFallback(),
            'web_service': WebServiceFallback(),
            'image_processing': ImageProcessingFallback(),
            'database': DatabaseFallback()
        }
    
    def get_fallback(self, category: str) -> Any:
        """Obtiene una implementación de fallback por categoría"""
        return self.fallbacks.get(category)
    
    def register_fallback(self, category: str, implementation: Any):
        """Registra una nueva implementación de fallback"""
        self.fallbacks[category] = implementation
    
    def list_available_fallbacks(self) -> List[str]:
        """Lista las categorías de fallback disponibles"""
        return list(self.fallbacks.keys())


# Instancia global del registro de fallbacks
fallback_registry = FallbackRegistry()


def get_fallback(category: str) -> Any:
    """Función de conveniencia para obtener fallbacks"""
    return fallback_registry.get_fallback(category)


def create_robust_import(package_name: str, fallback_category: Optional[str] = None):
    """
    Crea una función de importación robusta con fallback
    
    Args:
        package_name: Nombre del paquete a importar
        fallback_category: Categoría de fallback a usar
    
    Returns:
        Función que importa el paquete o devuelve el fallback
    """
    def robust_import():
        try:
            import importlib
            return importlib.import_module(package_name)
        except ImportError:
            if fallback_category:
                fallback = get_fallback(fallback_category)
                if fallback:
                    warnings.warn(f"Usando fallback para {package_name}")
                    return fallback
            
            raise ImportError(f"Paquete {package_name} no disponible y sin fallback")
    
    return robust_import


if __name__ == "__main__":
    # Pruebas básicas de fallbacks
    print("🧪 Probando implementaciones de fallback...")
    
    # Probar fallback de deep learning
    dl_fallback = get_fallback('deep_learning')
    if dl_fallback:
        print("✅ Fallback de Deep Learning disponible")
        
        # Crear datos de prueba
        test_image = np.random.rand(100, 100)
        features = dl_fallback.extract_features(test_image)
        print(f"  Características extraídas: {len(features)} dimensiones")
    
    # Probar fallback de base de datos
    db_fallback = get_fallback('database')
    if db_fallback:
        print("✅ Fallback de Base de Datos disponible")
        
        # Crear índice de prueba
        index = db_fallback.create_vector_index(128)
        test_vectors = np.random.rand(10, 128)
        index.add(test_vectors)
        
        query = np.random.rand(128)
        distances, indices = index.search(query, k=3)
        print(f"  Búsqueda vectorial: encontrados {len(indices[0])} resultados")
    
    print("🎉 Pruebas de fallback completadas")