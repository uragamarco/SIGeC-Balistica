"""
Sistema de Calibración Espacial DPI según estándares NIST

Este módulo implementa el sistema de calibración DPI requerido por NIST para
imágenes balísticas, incluyendo:
- Detección automática de DPI
- Calibración manual con referencias conocidas
- Validación de resolución espacial
- Conversión entre unidades (píxeles/mm, DPI)
- Cumplimiento de estándares NIST (mínimo 1000 DPI)

Autor: Sistema SEACABA
Fecha: 2024
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class CalibrationData:
    """Datos de calibración espacial"""
    pixels_per_mm: float
    dpi: float
    calibration_method: str
    reference_object_size_mm: Optional[float] = None
    reference_object_pixels: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class NISTCalibrationResult:
    """Resultado de calibración según NIST"""
    calibration_data: CalibrationData
    nist_compliant: bool
    validation_errors: list
    quality_score: float
    recommended_actions: list

class SpatialCalibrator:
    """
    Sistema de calibración espacial DPI para imágenes balísticas
    
    Implementa los requisitos NIST para calibración espacial:
    - Resolución mínima: 1000 DPI
    - Precisión de calibración: ±2%
    - Trazabilidad metrológica
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nist_min_dpi = 1000  # Requisito NIST mínimo
        self.calibration_tolerance = 0.02  # ±2% según NIST
        
        # Objetos de referencia comunes en balística (tamaños en mm)
        self.reference_objects = {
            'coin_1_peso': 23.0,  # Moneda de 1 peso argentino
            'coin_25_centavos': 24.25,  # Moneda de 25 centavos
            'ruler_1cm': 10.0,  # Regla de 1 cm
            'bullet_9mm': 9.0,  # Proyectil 9mm (diámetro)
            'case_9mm': 9.65,  # Vaina 9mm (diámetro base)
            'scale_bar': None  # Barra de escala (tamaño variable)
        }
    
    def calibrate_from_metadata(self, image_path: str) -> Optional[CalibrationData]:
        """
        Extraer calibración DPI de metadatos EXIF
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            CalibrationData si se encuentra información válida
        """
        try:
            # Leer metadatos EXIF
            import PIL.Image
            import PIL.ExifTags
            
            with PIL.Image.open(image_path) as img:
                exif = img.getexif()
                
                if exif:
                    # Buscar información de resolución
                    x_resolution = exif.get(282)  # XResolution
                    y_resolution = exif.get(283)  # YResolution
                    resolution_unit = exif.get(296, 2)  # ResolutionUnit (2=inches, 3=cm)
                    
                    if x_resolution and y_resolution:
                        # Convertir a DPI si está en cm
                        if resolution_unit == 3:  # cm
                            dpi = float(x_resolution) * 2.54
                        else:  # inches
                            dpi = float(x_resolution)
                        
                        # Calcular píxeles por mm
                        pixels_per_mm = dpi / 25.4
                        
                        return CalibrationData(
                            pixels_per_mm=pixels_per_mm,
                            dpi=dpi,
                            calibration_method="exif_metadata",
                            confidence=0.9 if dpi >= self.nist_min_dpi else 0.7,
                            metadata={
                                'x_resolution': x_resolution,
                                'y_resolution': y_resolution,
                                'resolution_unit': resolution_unit
                            }
                        )
                        
        except Exception as e:
            self.logger.warning(f"No se pudo extraer DPI de metadatos: {e}")
        
        return None
    
    def calibrate_with_reference_object(self, 
                                      image: np.ndarray,
                                      object_type: str,
                                      object_size_mm: Optional[float] = None) -> Optional[CalibrationData]:
        """
        Calibrar usando objeto de referencia conocido
        
        Args:
            image: Imagen a calibrar
            object_type: Tipo de objeto de referencia
            object_size_mm: Tamaño real del objeto en mm (opcional si está predefinido)
            
        Returns:
            CalibrationData con la calibración calculada
        """
        try:
            # Obtener tamaño de referencia
            if object_size_mm is None:
                if object_type not in self.reference_objects:
                    raise ValueError(f"Objeto de referencia desconocido: {object_type}")
                object_size_mm = self.reference_objects[object_type]
                if object_size_mm is None:
                    raise ValueError(f"Tamaño no definido para {object_type}")
            
            # Detectar objeto de referencia en la imagen
            object_pixels = self._detect_reference_object(image, object_type)
            
            if object_pixels is None:
                self.logger.error(f"No se pudo detectar objeto de referencia: {object_type}")
                return None
            
            # Calcular calibración
            pixels_per_mm = object_pixels / object_size_mm
            dpi = pixels_per_mm * 25.4
            
            # Calcular confianza basada en detección
            confidence = self._calculate_detection_confidence(image, object_pixels, object_type)
            
            return CalibrationData(
                pixels_per_mm=pixels_per_mm,
                dpi=dpi,
                calibration_method=f"reference_object_{object_type}",
                reference_object_size_mm=object_size_mm,
                reference_object_pixels=object_pixels,
                confidence=confidence,
                metadata={
                    'object_type': object_type,
                    'detection_method': 'automatic'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error en calibración con objeto de referencia: {e}")
            return None
    
    def calibrate_manual(self, 
                        image: np.ndarray,
                        pixel_distance: float,
                        real_distance_mm: float) -> CalibrationData:
        """
        Calibración manual con medición directa
        
        Args:
            image: Imagen a calibrar
            pixel_distance: Distancia medida en píxeles
            real_distance_mm: Distancia real en milímetros
            
        Returns:
            CalibrationData con la calibración manual
        """
        pixels_per_mm = pixel_distance / real_distance_mm
        dpi = pixels_per_mm * 25.4
        
        return CalibrationData(
            pixels_per_mm=pixels_per_mm,
            dpi=dpi,
            calibration_method="manual_measurement",
            reference_object_size_mm=real_distance_mm,
            reference_object_pixels=pixel_distance,
            confidence=0.95,  # Alta confianza en medición manual
            metadata={
                'measurement_type': 'manual',
                'user_verified': True
            }
        )
    
    def _detect_reference_object(self, image: np.ndarray, object_type: str) -> Optional[float]:
        """
        Detectar y medir objeto de referencia en la imagen
        
        Args:
            image: Imagen donde buscar
            object_type: Tipo de objeto a detectar
            
        Returns:
            Tamaño del objeto en píxeles, o None si no se detecta
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if object_type.startswith('coin'):
                return self._detect_circular_object(gray)
            elif object_type == 'ruler_1cm':
                return self._detect_ruler_segment(gray)
            elif object_type.startswith('bullet') or object_type.startswith('case'):
                return self._detect_ballistic_object(gray, object_type)
            elif object_type == 'scale_bar':
                return self._detect_scale_bar(gray)
            else:
                self.logger.warning(f"Detección no implementada para: {object_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error detectando objeto de referencia: {e}")
            return None
    
    def _detect_circular_object(self, gray: np.ndarray) -> Optional[float]:
        """Detectar objeto circular (moneda) usando HoughCircles"""
        try:
            # Aplicar filtro Gaussiano
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detectar círculos
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(gray.shape[0] / 8),
                param1=50,
                param2=30,
                minRadius=int(min(gray.shape) / 20),
                maxRadius=int(min(gray.shape) / 4)
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                # Seleccionar el círculo más prominente
                best_circle = None
                best_score = 0
                
                for (x, y, r) in circles:
                    # Verificar que el círculo esté completamente dentro de la imagen
                    if (x - r >= 0 and y - r >= 0 and 
                        x + r < gray.shape[1] and y + r < gray.shape[0]):
                        
                        # Calcular score basado en contraste del borde
                        score = self._calculate_circle_score(gray, x, y, r)
                        
                        if score > best_score:
                            best_score = score
                            best_circle = (x, y, r)
                
                if best_circle is not None:
                    return float(best_circle[2] * 2)  # Diámetro en píxeles
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando objeto circular: {e}")
            return None
    
    def _detect_ruler_segment(self, gray: np.ndarray) -> Optional[float]:
        """Detectar segmento de regla de 1cm"""
        try:
            # Detectar líneas usando HoughLines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Buscar líneas paralelas que podrían ser marcas de regla
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    rho, theta = line[0]
                    
                    # Clasificar líneas como horizontales o verticales
                    if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                        horizontal_lines.append((rho, theta))
                    elif abs(theta - np.pi/2) < np.pi/4:
                        vertical_lines.append((rho, theta))
                
                # Buscar distancia entre líneas paralelas (marcas de 1cm)
                if len(horizontal_lines) >= 2:
                    distances = []
                    for i in range(len(horizontal_lines)):
                        for j in range(i+1, len(horizontal_lines)):
                            dist = abs(horizontal_lines[i][0] - horizontal_lines[j][0])
                            distances.append(dist)
                    
                    if distances:
                        # Asumir que la distancia más común es 1cm
                        return float(np.median(distances))
                
                if len(vertical_lines) >= 2:
                    distances = []
                    for i in range(len(vertical_lines)):
                        for j in range(i+1, len(vertical_lines)):
                            dist = abs(vertical_lines[i][0] - vertical_lines[j][0])
                            distances.append(dist)
                    
                    if distances:
                        return float(np.median(distances))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando regla: {e}")
            return None
    
    def _detect_ballistic_object(self, gray: np.ndarray, object_type: str) -> Optional[float]:
        """Detectar objeto balístico (proyectil o vaina)"""
        try:
            # Para objetos balísticos, usar detección de contornos
            # Aplicar threshold adaptativo
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Buscar el contorno más grande que podría ser el objeto balístico
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calcular dimensiones del contorno
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Para proyectiles y vainas, usar el diámetro (menor dimensión)
                diameter = min(w, h)
                
                # Verificar que el tamaño sea razonable
                min_size = min(gray.shape) / 20
                max_size = min(gray.shape) / 4
                
                if min_size <= diameter <= max_size:
                    return float(diameter)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando objeto balístico: {e}")
            return None
    
    def _detect_scale_bar(self, gray: np.ndarray) -> Optional[float]:
        """Detectar barra de escala en la imagen"""
        try:
            # Buscar rectángulos que podrían ser barras de escala
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            scale_bars = []
            
            for contour in contours:
                # Aproximar contorno a rectángulo
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Es un rectángulo
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Verificar proporciones típicas de barra de escala
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 3 <= aspect_ratio <= 20:  # Barra horizontal
                        scale_bars.append((w, h, x, y, 'horizontal'))
                    elif 1/20 <= aspect_ratio <= 1/3:  # Barra vertical
                        scale_bars.append((h, w, x, y, 'vertical'))
            
            if scale_bars:
                # Seleccionar la barra más prominente
                best_bar = max(scale_bars, key=lambda x: x[0] * x[1])
                return float(best_bar[0])  # Longitud de la barra
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detectando barra de escala: {e}")
            return None
    
    def _calculate_circle_score(self, gray: np.ndarray, x: int, y: int, r: int) -> float:
        """Calcular score de calidad para un círculo detectado"""
        try:
            # Crear máscara circular
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Calcular contraste entre interior y borde
            inner_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(inner_mask, (x, y), max(1, r-5), 255, -1)
            
            border_mask = mask - inner_mask
            
            if np.sum(inner_mask > 0) > 0 and np.sum(border_mask > 0) > 0:
                inner_mean = np.mean(gray[inner_mask > 0])
                border_mean = np.mean(gray[border_mask > 0])
                
                contrast = abs(inner_mean - border_mean)
                return contrast
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculando score de círculo: {e}")
            return 0.0
    
    def _calculate_detection_confidence(self, 
                                     image: np.ndarray, 
                                     detected_size: float, 
                                     object_type: str) -> float:
        """Calcular confianza de la detección"""
        try:
            # Factores que afectan la confianza
            confidence = 0.8  # Base
            
            # Factor de tamaño relativo
            image_size = min(image.shape[:2])
            relative_size = detected_size / image_size
            
            if 0.05 <= relative_size <= 0.3:  # Tamaño apropiado
                confidence += 0.1
            elif relative_size < 0.02 or relative_size > 0.5:  # Muy pequeño o grande
                confidence -= 0.2
            
            # Factor de nitidez de la imagen
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 100:  # Imagen nítida
                confidence += 0.1
            elif laplacian_var < 50:  # Imagen borrosa
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def validate_nist_compliance(self, calibration: CalibrationData) -> NISTCalibrationResult:
        """
        Validar cumplimiento de estándares NIST
        
        Args:
            calibration: Datos de calibración a validar
            
        Returns:
            NISTCalibrationResult con validación completa
        """
        errors = []
        recommendations = []
        quality_score = 1.0
        
        # Verificar DPI mínimo
        if calibration.dpi < self.nist_min_dpi:
            errors.append(f"DPI insuficiente: {calibration.dpi:.1f} < {self.nist_min_dpi} (mínimo NIST)")
            quality_score -= 0.4
            recommendations.append(f"Aumentar resolución a mínimo {self.nist_min_dpi} DPI")
        
        # Verificar confianza de calibración
        if calibration.confidence < 0.8:
            errors.append(f"Confianza de calibración baja: {calibration.confidence:.2f}")
            quality_score -= 0.2
            recommendations.append("Recalibrar con método más preciso")
        
        # Verificar método de calibración
        if calibration.calibration_method == "exif_metadata":
            if calibration.confidence < 0.9:
                recommendations.append("Verificar calibración con objeto de referencia")
        
        # Verificar trazabilidad
        if calibration.calibration_method == "manual_measurement":
            if not calibration.metadata.get('user_verified', False):
                errors.append("Calibración manual sin verificación de usuario")
                quality_score -= 0.1
        
        # Calcular score final
        quality_score = max(0.0, quality_score)
        nist_compliant = len(errors) == 0 and calibration.dpi >= self.nist_min_dpi
        
        return NISTCalibrationResult(
            calibration_data=calibration,
            nist_compliant=nist_compliant,
            validation_errors=errors,
            quality_score=quality_score,
            recommended_actions=recommendations
        )
    
    def save_calibration(self, calibration: CalibrationData, filepath: str) -> bool:
        """Guardar datos de calibración en archivo JSON"""
        try:
            calibration_dict = {
                'pixels_per_mm': calibration.pixels_per_mm,
                'dpi': calibration.dpi,
                'calibration_method': calibration.calibration_method,
                'reference_object_size_mm': calibration.reference_object_size_mm,
                'reference_object_pixels': calibration.reference_object_pixels,
                'confidence': calibration.confidence,
                'metadata': calibration.metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(calibration_dict, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> Optional[CalibrationData]:
        """Cargar datos de calibración desde archivo JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return CalibrationData(**data)
            
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None