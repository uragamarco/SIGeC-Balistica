"""
Métricas de Calidad NIST para Análisis Balístico
===============================================

Este módulo implementa las métricas de calidad de imagen según los estándares NIST
para análisis forense balístico, incluyendo:

- Signal-to-Noise Ratio (SNR)
- Contraste según estándares NIST
- Uniformidad de iluminación
- Métricas de nitidez y resolución
- Evaluación de calidad global

Basado en:
- NIST Special Publication 800-101 Rev. 1
- ISO/IEC 19794-4 (Biometric Data Interchange Formats)
- ANSI/NIST-ITL 1-2011 Update 2015
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math
from scipy import ndimage
from skimage import measure, filters, feature
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings('ignore')


class QualityLevel(Enum):
    """Niveles de calidad según NIST"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class NISTQualityReport:
    """Reporte de calidad según estándares NIST"""
    image_id: str
    overall_quality: QualityLevel
    snr_value: float
    contrast_value: float
    uniformity_value: float
    sharpness_value: float
    resolution_value: float
    noise_level: float
    brightness_level: float
    saturation_level: float
    quality_score: float
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el reporte a diccionario"""
        data = {
            'image_id': self.image_id,
            'overall_quality': self.overall_quality.value,
            'snr_value': self.snr_value,
            'contrast_value': self.contrast_value,
            'uniformity_value': self.uniformity_value,
            'sharpness_value': self.sharpness_value,
            'resolution_value': self.resolution_value,
            'noise_level': self.noise_level,
            'brightness_level': self.brightness_level,
            'saturation_level': self.saturation_level,
            'quality_score': self.quality_score,
            'recommendations': self.recommendations,
            'detailed_metrics': self.detailed_metrics
        }
        return data


class NISTQualityMetrics:
    """
    Implementación de métricas de calidad según estándares NIST
    """
    
    def __init__(self):
        # Umbrales según estándares NIST
        self.snr_thresholds = {
            QualityLevel.EXCELLENT: 30.0,
            QualityLevel.GOOD: 25.0,
            QualityLevel.FAIR: 20.0,
            QualityLevel.POOR: 15.0,
            QualityLevel.UNACCEPTABLE: 0.0
        }
        
        self.contrast_thresholds = {
            QualityLevel.EXCELLENT: 0.8,
            QualityLevel.GOOD: 0.6,
            QualityLevel.FAIR: 0.4,
            QualityLevel.POOR: 0.2,
            QualityLevel.UNACCEPTABLE: 0.0
        }
        
        self.uniformity_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.8,
            QualityLevel.FAIR: 0.7,
            QualityLevel.POOR: 0.6,
            QualityLevel.UNACCEPTABLE: 0.0
        }
        
        self.sharpness_thresholds = {
            QualityLevel.EXCELLENT: 0.8,
            QualityLevel.GOOD: 0.6,
            QualityLevel.FAIR: 0.4,
            QualityLevel.POOR: 0.2,
            QualityLevel.UNACCEPTABLE: 0.0
        }
    
    def analyze_image_quality(self, image: np.ndarray, image_id: str = "unknown") -> NISTQualityReport:
        """
        Analiza la calidad de una imagen según estándares NIST
        
        Args:
            image: Imagen a analizar (numpy array)
            image_id: Identificador de la imagen
            
        Returns:
            NISTQualityReport: Reporte completo de calidad
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Calcular métricas individuales
        snr_value = self.calculate_snr(gray_image)
        contrast_value = self.calculate_contrast_nist(gray_image)
        uniformity_value = self.calculate_uniformity(gray_image)
        sharpness_value = self.calculate_sharpness(gray_image)
        resolution_value = self.calculate_resolution_metric(gray_image)
        noise_level = self.calculate_noise_level(gray_image)
        brightness_level = self.calculate_brightness_level(gray_image)
        saturation_level = self.calculate_saturation_level(image)
        
        # Métricas detalladas
        detailed_metrics = self._calculate_detailed_metrics(gray_image, image)
        
        # Calcular puntuación global
        quality_score = self._calculate_overall_quality_score(
            snr_value, contrast_value, uniformity_value, sharpness_value
        )
        
        # Determinar nivel de calidad global
        overall_quality = self._determine_overall_quality(
            snr_value, contrast_value, uniformity_value, sharpness_value
        )
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            snr_value, contrast_value, uniformity_value, sharpness_value,
            noise_level, brightness_level
        )
        
        return NISTQualityReport(
            image_id=image_id,
            overall_quality=overall_quality,
            snr_value=snr_value,
            contrast_value=contrast_value,
            uniformity_value=uniformity_value,
            sharpness_value=sharpness_value,
            resolution_value=resolution_value,
            noise_level=noise_level,
            brightness_level=brightness_level,
            saturation_level=saturation_level,
            quality_score=quality_score,
            recommendations=recommendations,
            detailed_metrics=detailed_metrics
        )
    
    def calculate_snr(self, image: np.ndarray) -> float:
        """
        Calcula Signal-to-Noise Ratio según estándares NIST
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Valor SNR en dB
        """
        try:
            # Método 1: SNR basado en varianza local
            # Aplicar filtro Gaussiano para obtener señal suavizada
            signal = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.0)
            
            # Calcular ruido como diferencia entre imagen original y señal
            noise = image.astype(np.float32) - signal
            
            # Calcular potencia de señal y ruido
            signal_power = np.mean(signal ** 2)
            noise_power = np.mean(noise ** 2)
            
            # Evitar división por cero
            if noise_power == 0:
                return 100.0  # SNR muy alto
            
            # Calcular SNR en dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            
            # Método 2: SNR basado en regiones homogéneas (validación)
            snr_homogeneous = self._calculate_snr_homogeneous_regions(image)
            
            # Usar el promedio de ambos métodos
            snr_final = (snr_db + snr_homogeneous) / 2.0
            
            return max(0.0, min(100.0, snr_final))
            
        except Exception as e:
            print(f"Error calculando SNR: {e}")
            return 0.0
    
    def _calculate_snr_homogeneous_regions(self, image: np.ndarray) -> float:
        """
        Calcula SNR en regiones homogéneas de la imagen
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Valor SNR en dB
        """
        try:
            # Dividir imagen en bloques
            h, w = image.shape
            block_size = min(32, h // 8, w // 8)
            
            if block_size < 8:
                return self.calculate_snr(image)
            
            snr_values = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    
                    # Calcular varianza del bloque
                    block_var = np.var(block)
                    
                    # Solo considerar bloques con varianza baja (homogéneos)
                    if block_var < np.var(image) * 0.5:
                        mean_val = np.mean(block)
                        if mean_val > 0:
                            snr_block = 20 * np.log10(mean_val / (np.sqrt(block_var) + 1e-10))
                            snr_values.append(snr_block)
            
            if snr_values:
                return np.mean(snr_values)
            else:
                return 20.0  # Valor por defecto
                
        except Exception:
            return 20.0
    
    def calculate_contrast_nist(self, image: np.ndarray) -> float:
        """
        Calcula contraste según estándares NIST
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Valor de contraste normalizado (0-1)
        """
        try:
            # Método 1: Contraste RMS (Root Mean Square)
            mean_intensity = np.mean(image)
            rms_contrast = np.sqrt(np.mean((image - mean_intensity) ** 2)) / 255.0
            
            # Método 2: Contraste Michelson
            max_intensity = np.max(image)
            min_intensity = np.min(image)
            
            if max_intensity + min_intensity > 0:
                michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
            else:
                michelson_contrast = 0.0
            
            # Método 3: Contraste basado en gradientes
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_contrast = np.mean(gradient_magnitude) / 255.0
            
            # Combinar métodos con pesos
            contrast_final = (0.4 * rms_contrast + 
                            0.3 * michelson_contrast + 
                            0.3 * gradient_contrast)
            
            return max(0.0, min(1.0, contrast_final))
            
        except Exception as e:
            print(f"Error calculando contraste: {e}")
            return 0.0
    
    def calculate_uniformity(self, image: np.ndarray) -> float:
        """
        Calcula uniformidad de iluminación según NIST
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Valor de uniformidad (0-1)
        """
        try:
            # Método 1: Uniformidad basada en varianza local
            # Dividir imagen en regiones
            h, w = image.shape
            region_size = min(64, h // 4, w // 4)
            
            if region_size < 16:
                return 0.5  # Imagen muy pequeña
            
            region_means = []
            
            for i in range(0, h - region_size, region_size // 2):
                for j in range(0, w - region_size, region_size // 2):
                    region = image[i:i+region_size, j:j+region_size]
                    region_means.append(np.mean(region))
            
            if len(region_means) < 2:
                return 0.5
            
            # Calcular uniformidad como inverso de la varianza normalizada
            mean_of_means = np.mean(region_means)
            var_of_means = np.var(region_means)
            
            if mean_of_means > 0:
                uniformity_local = 1.0 - (np.sqrt(var_of_means) / mean_of_means)
            else:
                uniformity_local = 0.0
            
            # Método 2: Uniformidad basada en histograma
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_normalized = hist / np.sum(hist)
            
            # Calcular entropía del histograma
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            max_entropy = np.log2(256)  # Entropía máxima para 8 bits
            uniformity_hist = entropy / max_entropy
            
            # Combinar ambos métodos
            uniformity_final = 0.7 * max(0.0, uniformity_local) + 0.3 * uniformity_hist
            
            return max(0.0, min(1.0, uniformity_final))
            
        except Exception as e:
            print(f"Error calculando uniformidad: {e}")
            return 0.0
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calcula nitidez de la imagen
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Valor de nitidez normalizado (0-1)
        """
        try:
            # Método 1: Varianza del Laplaciano
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            sharpness_laplacian = min(1.0, laplacian_var / 1000.0)
            
            # Método 2: Gradiente promedio
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            sharpness_gradient = min(1.0, np.mean(gradient_magnitude) / 100.0)
            
            # Método 3: Análisis de frecuencias (FFT)
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calcular energía en altas frecuencias
            h, w = image.shape
            center_h, center_w = h // 2, w // 2
            
            # Crear máscara para altas frecuencias
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) // 4)**2
            
            high_freq_energy = np.sum(magnitude_spectrum[mask])
            total_energy = np.sum(magnitude_spectrum)
            
            if total_energy > 0:
                sharpness_freq = high_freq_energy / total_energy
            else:
                sharpness_freq = 0.0
            
            # Combinar métodos
            sharpness_final = (0.4 * sharpness_laplacian + 
                             0.4 * sharpness_gradient + 
                             0.2 * sharpness_freq)
            
            return max(0.0, min(1.0, sharpness_final))
            
        except Exception as e:
            print(f"Error calculando nitidez: {e}")
            return 0.0
    
    def calculate_resolution_metric(self, image: np.ndarray) -> float:
        """
        Calcula métrica de resolución efectiva
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Métrica de resolución (0-1)
        """
        try:
            # Análisis de resolución basado en MTF (Modulation Transfer Function)
            # Simplificado usando análisis de bordes
            
            # Detectar bordes usando Canny
            edges = cv2.Canny(image, 50, 150)
            
            # Calcular densidad de bordes
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            
            # Normalizar densidad de bordes
            resolution_metric = min(1.0, edge_density * 10.0)
            
            return max(0.0, resolution_metric)
            
        except Exception as e:
            print(f"Error calculando resolución: {e}")
            return 0.0
    
    def calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Calcula nivel de ruido en la imagen
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Nivel de ruido normalizado (0-1)
        """
        try:
            # Método basado en filtro de mediana
            median_filtered = cv2.medianBlur(image, 5)
            noise = image.astype(np.float32) - median_filtered.astype(np.float32)
            noise_level = np.std(noise) / 255.0
            
            return max(0.0, min(1.0, noise_level))
            
        except Exception as e:
            print(f"Error calculando nivel de ruido: {e}")
            return 0.0
    
    def calculate_brightness_level(self, image: np.ndarray) -> float:
        """
        Calcula nivel de brillo de la imagen
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            float: Nivel de brillo normalizado (0-1)
        """
        try:
            brightness = np.mean(image) / 255.0
            return max(0.0, min(1.0, brightness))
            
        except Exception as e:
            print(f"Error calculando brillo: {e}")
            return 0.0
    
    def calculate_saturation_level(self, image: np.ndarray) -> float:
        """
        Calcula nivel de saturación (para imágenes en color)
        
        Args:
            image: Imagen (puede ser color o escala de grises)
            
        Returns:
            float: Nivel de saturación normalizado (0-1)
        """
        try:
            if len(image.shape) == 3:
                # Convertir a HSV para obtener saturación
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                saturation = np.mean(hsv[:, :, 1]) / 255.0
                return max(0.0, min(1.0, saturation))
            else:
                return 0.0  # Imagen en escala de grises
                
        except Exception as e:
            print(f"Error calculando saturación: {e}")
            return 0.0
    
    def _calculate_detailed_metrics(self, gray_image: np.ndarray, 
                                  original_image: np.ndarray) -> Dict[str, Any]:
        """
        Calcula métricas detalladas adicionales
        
        Args:
            gray_image: Imagen en escala de grises
            original_image: Imagen original
            
        Returns:
            Dict: Métricas detalladas
        """
        try:
            metrics = {}
            
            # Estadísticas básicas
            metrics['mean_intensity'] = float(np.mean(gray_image))
            metrics['std_intensity'] = float(np.std(gray_image))
            metrics['min_intensity'] = int(np.min(gray_image))
            metrics['max_intensity'] = int(np.max(gray_image))
            
            # Histograma
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            metrics['histogram_entropy'] = float(-np.sum((hist/np.sum(hist)) * np.log2((hist/np.sum(hist)) + 1e-10)))
            
            # Análisis de textura
            try:
                # LBP (Local Binary Pattern)
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
                metrics['texture_uniformity'] = float(np.var(lbp))
            except ImportError:
                metrics['texture_uniformity'] = 0.0
            
            # Análisis de frecuencias
            f_transform = np.fft.fft2(gray_image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            metrics['frequency_content'] = float(np.mean(magnitude_spectrum))
            
            # Métricas de forma (si hay bordes detectables)
            edges = cv2.Canny(gray_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            metrics['edge_count'] = len(contours)
            
            if contours:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                if areas:
                    metrics['avg_contour_area'] = float(np.mean(areas))
                    metrics['max_contour_area'] = float(np.max(areas))
                else:
                    metrics['avg_contour_area'] = 0.0
                    metrics['max_contour_area'] = 0.0
            else:
                metrics['avg_contour_area'] = 0.0
                metrics['max_contour_area'] = 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculando métricas detalladas: {e}")
            return {}
    
    def _calculate_overall_quality_score(self, snr: float, contrast: float, 
                                       uniformity: float, sharpness: float) -> float:
        """
        Calcula puntuación global de calidad
        
        Args:
            snr: Valor SNR
            contrast: Valor de contraste
            uniformity: Valor de uniformidad
            sharpness: Valor de nitidez
            
        Returns:
            float: Puntuación global (0-100)
        """
        try:
            # Normalizar SNR a escala 0-1
            snr_normalized = min(1.0, max(0.0, snr / 40.0))
            
            # Pesos para cada métrica
            weights = {
                'snr': 0.3,
                'contrast': 0.25,
                'uniformity': 0.25,
                'sharpness': 0.2
            }
            
            # Calcular puntuación ponderada
            score = (weights['snr'] * snr_normalized +
                    weights['contrast'] * contrast +
                    weights['uniformity'] * uniformity +
                    weights['sharpness'] * sharpness)
            
            return score * 100.0
            
        except Exception:
            return 0.0
    
    def _determine_overall_quality(self, snr: float, contrast: float, 
                                 uniformity: float, sharpness: float) -> QualityLevel:
        """
        Determina el nivel de calidad global
        
        Args:
            snr: Valor SNR
            contrast: Valor de contraste
            uniformity: Valor de uniformidad
            sharpness: Valor de nitidez
            
        Returns:
            QualityLevel: Nivel de calidad determinado
        """
        try:
            # Contar métricas que cumplen cada umbral
            quality_counts = {level: 0 for level in QualityLevel}
            
            # Evaluar SNR
            for level in QualityLevel:
                if snr >= self.snr_thresholds[level]:
                    quality_counts[level] += 1
                    break
            
            # Evaluar contraste
            for level in QualityLevel:
                if contrast >= self.contrast_thresholds[level]:
                    quality_counts[level] += 1
                    break
            
            # Evaluar uniformidad
            for level in QualityLevel:
                if uniformity >= self.uniformity_thresholds[level]:
                    quality_counts[level] += 1
                    break
            
            # Evaluar nitidez
            for level in QualityLevel:
                if sharpness >= self.sharpness_thresholds[level]:
                    quality_counts[level] += 1
                    break
            
            # Determinar nivel basado en mayoría
            total_metrics = 4
            
            if quality_counts[QualityLevel.EXCELLENT] >= total_metrics * 0.75:
                return QualityLevel.EXCELLENT
            elif quality_counts[QualityLevel.GOOD] >= total_metrics * 0.5:
                return QualityLevel.GOOD
            elif quality_counts[QualityLevel.FAIR] >= total_metrics * 0.25:
                return QualityLevel.FAIR
            elif quality_counts[QualityLevel.POOR] > 0:
                return QualityLevel.POOR
            else:
                return QualityLevel.UNACCEPTABLE
                
        except Exception:
            return QualityLevel.UNACCEPTABLE
    
    def _generate_recommendations(self, snr: float, contrast: float, 
                                uniformity: float, sharpness: float,
                                noise_level: float, brightness_level: float) -> List[str]:
        """
        Genera recomendaciones para mejorar la calidad
        
        Args:
            snr: Valor SNR
            contrast: Valor de contraste
            uniformity: Valor de uniformidad
            sharpness: Valor de nitidez
            noise_level: Nivel de ruido
            brightness_level: Nivel de brillo
            
        Returns:
            List[str]: Lista de recomendaciones
        """
        recommendations = []
        
        try:
            # Recomendaciones para SNR
            if snr < self.snr_thresholds[QualityLevel.FAIR]:
                recommendations.append("Mejorar relación señal-ruido: reducir ruido de adquisición")
                recommendations.append("Verificar configuración de cámara y condiciones de iluminación")
            
            # Recomendaciones para contraste
            if contrast < self.contrast_thresholds[QualityLevel.FAIR]:
                recommendations.append("Aumentar contraste: ajustar iluminación o configuración de cámara")
                recommendations.append("Considerar técnicas de mejora de contraste en post-procesamiento")
            
            # Recomendaciones para uniformidad
            if uniformity < self.uniformity_thresholds[QualityLevel.FAIR]:
                recommendations.append("Mejorar uniformidad de iluminación")
                recommendations.append("Verificar calibración del sistema de iluminación")
            
            # Recomendaciones para nitidez
            if sharpness < self.sharpness_thresholds[QualityLevel.FAIR]:
                recommendations.append("Mejorar enfoque: verificar calibración del sistema óptico")
                recommendations.append("Reducir vibración durante la adquisición")
            
            # Recomendaciones para ruido
            if noise_level > 0.3:
                recommendations.append("Reducir ruido: usar menor ISO o mayor tiempo de exposición")
                recommendations.append("Aplicar filtros de reducción de ruido apropiados")
            
            # Recomendaciones para brillo
            if brightness_level < 0.2:
                recommendations.append("Aumentar iluminación: imagen demasiado oscura")
            elif brightness_level > 0.8:
                recommendations.append("Reducir iluminación: imagen demasiado brillante")
            
            # Recomendaciones generales
            if len(recommendations) == 0:
                recommendations.append("Calidad de imagen aceptable para análisis")
            else:
                recommendations.append("Repetir adquisición con configuración mejorada si es posible")
            
            return recommendations
            
        except Exception:
            return ["Error generando recomendaciones"]
    
    def compare_quality_reports(self, report1: NISTQualityReport, 
                              report2: NISTQualityReport) -> Dict[str, Any]:
        """
        Compara dos reportes de calidad
        
        Args:
            report1: Primer reporte
            report2: Segundo reporte
            
        Returns:
            Dict: Comparación detallada
        """
        try:
            comparison = {
                'image1_id': report1.image_id,
                'image2_id': report2.image_id,
                'quality_comparison': {
                    'image1_quality': report1.overall_quality.value,
                    'image2_quality': report2.overall_quality.value,
                    'better_quality': report1.image_id if report1.quality_score > report2.quality_score else report2.image_id
                },
                'metric_differences': {
                    'snr_diff': report1.snr_value - report2.snr_value,
                    'contrast_diff': report1.contrast_value - report2.contrast_value,
                    'uniformity_diff': report1.uniformity_value - report2.uniformity_value,
                    'sharpness_diff': report1.sharpness_value - report2.sharpness_value,
                    'quality_score_diff': report1.quality_score - report2.quality_score
                },
                'recommendations': []
            }
            
            # Generar recomendaciones comparativas
            if report1.quality_score > report2.quality_score:
                comparison['recommendations'].append(f"Imagen {report1.image_id} tiene mejor calidad general")
            elif report2.quality_score > report1.quality_score:
                comparison['recommendations'].append(f"Imagen {report2.image_id} tiene mejor calidad general")
            else:
                comparison['recommendations'].append("Ambas imágenes tienen calidad similar")
            
            return comparison
            
        except Exception as e:
            return {'error': f"Error comparando reportes: {e}"}
    
    def batch_analyze_quality(self, images: List[Tuple[np.ndarray, str]]) -> List[NISTQualityReport]:
        """
        Analiza calidad de múltiples imágenes
        
        Args:
            images: Lista de tuplas (imagen, id)
            
        Returns:
            List[NISTQualityReport]: Lista de reportes de calidad
        """
        reports = []
        
        for image, image_id in images:
            try:
                report = self.analyze_image_quality(image, image_id)
                reports.append(report)
            except Exception as e:
                print(f"Error analizando imagen {image_id}: {e}")
                continue
        
        return reports
    
    def export_quality_report(self, report: NISTQualityReport, 
                            output_path: str, format: str = 'json') -> bool:
        """
        Exporta reporte de calidad a archivo
        
        Args:
            report: Reporte de calidad
            output_path: Ruta de salida
            format: Formato ('json' o 'xml')
            
        Returns:
            bool: True si se exportó correctamente
        """
        try:
            if format.lower() == 'json':
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'xml':
                # Implementar exportación XML si es necesario
                pass
            
            return True
            
        except Exception as e:
            print(f"Error exportando reporte: {e}")
            return False

    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Retorna los umbrales configurados para cada métrica (por nivel)."""
        def enum_dict_to_plain(d: Dict[QualityLevel, float]) -> Dict[str, float]:
            return {level.value: float(threshold) for level, threshold in d.items()}
        return {
            'snr': enum_dict_to_plain(self.snr_thresholds),
            'contrast': enum_dict_to_plain(self.contrast_thresholds),
            'uniformity': enum_dict_to_plain(self.uniformity_thresholds),
            'sharpness': enum_dict_to_plain(self.sharpness_thresholds)
        }

    def get_quality_score(self, report: Any) -> float:
        """Extrae puntaje de calidad global desde un reporte NIST o dict."""
        try:
            if hasattr(report, 'quality_score'):
                return float(getattr(report, 'quality_score'))
            if isinstance(report, dict) and 'quality_score' in report:
                return float(report.get('quality_score', 0.0))
        except Exception:
            pass
        return 0.0

    def meets_minimum_quality(self, report: Any, min_score: float) -> bool:
        """Verifica si el reporte cumple el puntaje mínimo de calidad."""
        try:
            return self.get_quality_score(report) >= float(min_score)
        except Exception:
            return False
