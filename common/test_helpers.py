#!/usr/bin/env python3
"""
Test Helpers - Utilidades para Tests
===================================

Este módulo proporciona utilidades comunes para los tests del sistema,
incluyendo generadores de imágenes de prueba y funciones auxiliares.

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import tempfile
import os


def create_test_image(width: int = 400, height: int = 400, 
                     pattern: str = "circles") -> np.ndarray:
    """
    Crea una imagen de prueba sintética con patrones específicos.
    
    Args:
        width: Ancho de la imagen
        height: Alto de la imagen
        pattern: Tipo de patrón ('circles', 'lines', 'ballistic')
        
    Returns:
        np.ndarray: Imagen de prueba
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    if pattern == "circles":
        # Patrón de círculos
        for i in range(5):
            for j in range(5):
                x = 50 + i * (width // 6)
                y = 50 + j * (height // 6)
                radius = 15 + (i + j) * 2
                cv2.circle(image, (x, y), radius, (255, 255, 255), -1)
                cv2.circle(image, (x, y), radius-5, (100, 100, 100), 2)
                
    elif pattern == "lines":
        # Patrón de líneas paralelas
        for i in range(10):
            y = 40 + i * (height // 12)
            cv2.line(image, (20, y), (width-20, y), (200, 200, 200), 2)
            
    elif pattern == "ballistic":
        # Patrón balístico simulado
        center = (width // 2, height // 2)
        
        # Círculo central (percutor)
        cv2.circle(image, center, 30, (200, 200, 200), -1)
        cv2.circle(image, center, 20, (100, 100, 100), -1)
        
        # Líneas radiales (estrías)
        for angle in range(0, 360, 30):
            x = int(center[0] + 80 * np.cos(np.radians(angle)))
            y = int(center[1] + 80 * np.sin(np.radians(angle)))
            cv2.line(image, center, (x, y), (150, 150, 150), 2)
    
    # Agregar ruido realista
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


class TestImageGenerator:
    """
    Generador de imágenes de prueba para evidencia balística.
    """
    
    def __init__(self):
        self.temp_dir = None
        
    def create_ballistic_image(self, 
                             width: int = 800, 
                             height: int = 600,
                             features: List[str] = None,
                             noise_level: float = 0.1,
                             rotation: float = 0.0,
                             blur_level: float = 0.0,
                             quality: str = 'high') -> np.ndarray:
        """
        Crea una imagen balística sintética con características específicas.
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            features: Lista de características a incluir
            noise_level: Nivel de ruido (0.0-1.0)
            rotation: Rotación en grados
            blur_level: Nivel de desenfoque
            quality: Calidad de la imagen ('high', 'medium', 'low')
            
        Returns:
            np.ndarray: Imagen balística sintética
        """
        if features is None:
            features = ['striations', 'firing_pin']
            
        # Crear imagen base
        image = np.ones((height, width, 3), dtype=np.uint8) * 120
        
        center = (width // 2, height // 2)
        
        # Agregar características según la lista
        if 'firing_pin' in features:
            self._add_firing_pin_mark(image, center)
            
        if 'striations' in features:
            self._add_striations(image, center, rotation)
            
        if 'breech_face' in features:
            self._add_breech_face_marks(image, center)
            
        if 'different_striations' in features:
            self._add_different_striations(image, center, rotation)
            
        if 'different_pin' in features:
            self._add_different_firing_pin(image, center)
            
        # Aplicar efectos según calidad
        if quality == 'low':
            noise_level *= 3
            blur_level += 1.0
        elif quality == 'medium':
            noise_level *= 1.5
            blur_level += 0.5
            
        # Agregar ruido
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 50, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        # Aplicar desenfoque
        if blur_level > 0:
            kernel_size = int(blur_level * 5) * 2 + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_level)
            
        return image
    
    def _add_firing_pin_mark(self, image: np.ndarray, center: Tuple[int, int]):
        """Agrega marca de percutor"""
        cv2.circle(image, center, 25, (180, 180, 180), -1)
        cv2.circle(image, center, 15, (100, 100, 100), -1)
        cv2.circle(image, center, 8, (80, 80, 80), -1)
        
    def _add_different_firing_pin(self, image: np.ndarray, center: Tuple[int, int]):
        """Agrega marca de percutor diferente"""
        # Forma rectangular en lugar de circular
        x, y = center
        cv2.rectangle(image, (x-20, y-15), (x+20, y+15), (180, 180, 180), -1)
        cv2.rectangle(image, (x-12, y-8), (x+12, y+8), (100, 100, 100), -1)
        
    def _add_striations(self, image: np.ndarray, center: Tuple[int, int], rotation: float = 0):
        """Agrega estrías"""
        for i in range(12):
            angle = i * 30 + rotation
            x1 = int(center[0] + 40 * np.cos(np.radians(angle)))
            y1 = int(center[1] + 40 * np.sin(np.radians(angle)))
            x2 = int(center[0] + 120 * np.cos(np.radians(angle)))
            y2 = int(center[1] + 120 * np.sin(np.radians(angle)))
            cv2.line(image, (x1, y1), (x2, y2), (160, 160, 160), 2)
            
    def _add_different_striations(self, image: np.ndarray, center: Tuple[int, int], rotation: float = 0):
        """Agrega estrías diferentes"""
        for i in range(8):  # Menos estrías
            angle = i * 45 + rotation + 22.5  # Ángulo diferente
            x1 = int(center[0] + 35 * np.cos(np.radians(angle)))
            y1 = int(center[1] + 35 * np.sin(np.radians(angle)))
            x2 = int(center[0] + 100 * np.cos(np.radians(angle)))
            y2 = int(center[1] + 100 * np.sin(np.radians(angle)))
            cv2.line(image, (x1, y1), (x2, y2), (140, 140, 140), 3)  # Más gruesas
            
    def _add_breech_face_marks(self, image: np.ndarray, center: Tuple[int, int]):
        """Agrega marcas de cara de recámara"""
        # Patrón de textura alrededor del centro
        for i in range(20):
            for j in range(20):
                x = center[0] - 100 + i * 10
                y = center[1] - 100 + j * 10
                if 50 < np.sqrt((x - center[0])**2 + (y - center[1])**2) < 150:
                    cv2.circle(image, (x, y), 2, (140, 140, 140), -1)
                    
    def save_image(self, image: np.ndarray, path: Path):
        """
        Guarda una imagen en el path especificado.
        
        Args:
            image: Imagen a guardar
            path: Ruta donde guardar la imagen
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        
    def create_test_dataset(self, 
                          num_images: int = 10,
                          output_dir: Optional[Path] = None) -> List[Path]:
        """
        Crea un dataset de imágenes de prueba.
        
        Args:
            num_images: Número de imágenes a crear
            output_dir: Directorio de salida
            
        Returns:
            List[Path]: Lista de rutas de las imágenes creadas
        """
        if output_dir is None:
            if self.temp_dir is None:
                self.temp_dir = Path(tempfile.mkdtemp())
            output_dir = self.temp_dir
            
        image_paths = []
        
        for i in range(num_images):
            # Variar características para cada imagen
            features = ['striations', 'firing_pin']
            if i % 3 == 0:
                features.append('breech_face')
                
            noise_level = 0.05 + (i % 4) * 0.05
            rotation = (i % 8) * 45
            
            image = self.create_ballistic_image(
                features=features,
                noise_level=noise_level,
                rotation=rotation
            )
            
            image_path = output_dir / f"test_image_{i:03d}.jpg"
            self.save_image(image, image_path)
            image_paths.append(image_path)
            
        return image_paths
        
    def cleanup(self):
        """Limpia archivos temporales"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


class MockDependencies:
    """
    Clase para crear mocks de dependencias en tests.
    """
    
    @staticmethod
    def create_mock_config() -> Dict[str, Any]:
        """Crea una configuración mock para tests"""
        return {
            'processing': {
                'max_workers': 2,
                'timeout': 30,
                'memory_limit_mb': 512
            },
            'matching': {
                'algorithm': 'cmc',
                'threshold': 0.7,
                'max_comparisons': 100
            },
            'database': {
                'type': 'sqlite',
                'path': ':memory:',
                'timeout': 10
            },
            'nist': {
                'compliance_level': 'basic',
                'validation_enabled': True
            }
        }
        
    @staticmethod
    def create_mock_pipeline_result() -> Dict[str, Any]:
        """Crea un resultado mock del pipeline"""
        return {
            'success': True,
            'similarity_score': 0.85,
            'match_confidence': 0.92,
            'processing_time': 1.23,
            'features_detected': 156,
            'quality_score': 0.88,
            'nist_compliant': True,
            'metadata': {
                'algorithm_version': '2.1.0',
                'timestamp': '2024-01-01T12:00:00Z'
            }
        }


# Funciones de utilidad adicionales
def setup_test_environment() -> Path:
    """
    Configura un entorno de pruebas temporal.
    
    Returns:
        Path: Directorio temporal creado
    """
    temp_dir = Path(tempfile.mkdtemp(prefix='sigec_test_'))
    
    # Crear subdirectorios estándar
    (temp_dir / 'images').mkdir()
    (temp_dir / 'results').mkdir()
    (temp_dir / 'cache').mkdir()
    
    return temp_dir


def cleanup_test_environment(temp_dir: Path):
    """
    Limpia el entorno de pruebas.
    
    Args:
        temp_dir: Directorio temporal a limpiar
    """
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)


def assert_image_valid(image: np.ndarray, 
                      min_width: int = 100, 
                      min_height: int = 100):
    """
    Verifica que una imagen sea válida.
    
    Args:
        image: Imagen a verificar
        min_width: Ancho mínimo
        min_height: Alto mínimo
        
    Raises:
        AssertionError: Si la imagen no es válida
    """
    assert isinstance(image, np.ndarray), "La imagen debe ser un numpy array"
    assert len(image.shape) in [2, 3], "La imagen debe ser 2D o 3D"
    assert image.shape[0] >= min_height, f"Alto mínimo: {min_height}"
    assert image.shape[1] >= min_width, f"Ancho mínimo: {min_width}"
    assert image.dtype == np.uint8, "La imagen debe ser uint8"


def compare_images_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula la similitud entre dos imágenes.
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        
    Returns:
        float: Similitud (0.0-1.0)
    """
    # Convertir a escala de grises si es necesario
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
    # Redimensionar si es necesario
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    # Calcular correlación normalizada
    correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
    return float(correlation.max())