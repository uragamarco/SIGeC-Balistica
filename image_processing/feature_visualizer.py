"""
Módulo de Visualización para Extracción de Características
Implementa visualizaciones para puntos clave SIFT/ORB y mapas de textura LBP/Gabor
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import logging
from pathlib import Path
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.measure import shannon_entropy
from .lbp_cache import cached_local_binary_pattern, get_lbp_cache_stats
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

class FeatureVisualizationError(Exception):
    """Excepción para errores de visualización de características"""
    pass

class FeatureVisualizer:
    """
    Clase para visualizar características extraídas de imágenes balísticas
    """
    
    def __init__(self, output_dir: str = "feature_visualizations"):
        """
        Inicializa el visualizador de características
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
    def visualize_keypoints(self, 
                          image: np.ndarray, 
                          keypoints: List[cv2.KeyPoint], 
                          algorithm: str = "SIFT",
                          save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualiza puntos clave SIFT u ORB sobre la imagen
        
        Args:
            image: Imagen original
            keypoints: Lista de puntos clave detectados
            algorithm: Algoritmo usado (SIFT, ORB, etc.)
            save_path: Ruta donde guardar la imagen (opcional)
            
        Returns:
            Imagen con puntos clave dibujados
        """
        try:
            # Validar entrada
            if image is None:
                raise FeatureVisualizationError("La imagen no puede ser None")
            
            if keypoints is None:
                keypoints = []
                logger.warning("No se proporcionaron keypoints, creando visualización sin puntos clave")
            
            # Convertir a color si es necesario
            if len(image.shape) == 2:
                img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_color = image.copy()
            
            # Dibujar puntos clave solo si existen
            if keypoints:
                if algorithm.upper() == "SIFT":
                    # Para SIFT, usar círculos con tamaño proporcional al tamaño del keypoint
                    img_with_keypoints = cv2.drawKeypoints(
                        img_color, keypoints, None, 
                        color=(0, 255, 0),  # Verde
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                else:  # ORB u otros
                    # Para ORB, usar círculos simples
                    img_with_keypoints = cv2.drawKeypoints(
                        img_color, keypoints, None, 
                        color=(255, 0, 0),  # Rojo
                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
                    )
            else:
                # Si no hay keypoints, usar la imagen original
                img_with_keypoints = img_color.copy()
            
            # Crear visualización mejorada con matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Imagen original
            if len(image.shape) == 2:
                axes[0].imshow(image, cmap='gray')
            else:
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Imagen Original')
            axes[0].axis('off')
            
            # Imagen con puntos clave
            axes[1].imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
            
            # Título dinámico basado en si hay keypoints
            if keypoints:
                axes[1].set_title(f'Puntos Clave {algorithm} ({len(keypoints)} detectados)')
                
                # Añadir información estadística
                sizes = [kp.size for kp in keypoints if hasattr(kp, 'size')]
                responses = [kp.response for kp in keypoints if hasattr(kp, 'response')]
                
                if sizes and responses:
                    fig.suptitle(
                        f'Análisis de Puntos Clave {algorithm}\n'
                        f'Tamaño promedio: {np.mean(sizes):.2f} ± {np.std(sizes):.2f}\n'
                        f'Respuesta promedio: {np.mean(responses):.4f} ± {np.std(responses):.4f}',
                        fontsize=12
                    )
            else:
                axes[1].set_title(f'No se detectaron puntos clave {algorithm}')
                axes[1].text(0.5, 0.5, 'No hay puntos clave\npara mostrar', 
                           transform=axes[1].transAxes, ha='center', va='center',
                           fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Guardar si se especifica ruta
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualización de puntos clave guardada en: {save_path}")
            
            plt.close()
            
            return img_with_keypoints
            
        except Exception as e:
            logger.error(f"Error visualizando puntos clave: {str(e)}")
            raise FeatureVisualizationError(f"Error en visualización de keypoints: {str(e)}")
    
    def visualize_lbp_texture(self, 
                            image: np.ndarray, 
                            radius: int = 3, 
                            n_points: int = 24,
                            save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualiza mapas de textura LBP (Local Binary Pattern)
        
        Args:
            image: Imagen en escala de grises
            radius: Radio para LBP
            n_points: Número de puntos de muestreo
            save_path: Ruta donde guardar la visualización
            
        Returns:
            Mapa de textura LBP
        """
        try:
            # Validar que la imagen no sea None
            if image is None:
                logger.error("La imagen proporcionada es None")
                raise FeatureVisualizationError("No se puede procesar una imagen None para visualización LBP")
            
            # Validar que la imagen sea un array de numpy válido
            if not isinstance(image, np.ndarray):
                logger.error(f"La imagen debe ser un array de numpy, recibido: {type(image)}")
                raise FeatureVisualizationError(f"Tipo de imagen inválido: {type(image)}")
            
            # Validar que la imagen tenga dimensiones válidas
            if len(image.shape) == 0 or image.size == 0:
                logger.error("La imagen está vacía o no tiene dimensiones válidas")
                raise FeatureVisualizationError("Imagen vacía o con dimensiones inválidas")
            
            # Asegurar que la imagen esté en escala de grises
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image.copy()
            
            # Calcular LBP usando cache
            lbp, _ = cached_local_binary_pattern(gray_img, n_points, radius, method='uniform')
            
            # Crear visualización
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Imagen original
            axes[0, 0].imshow(gray_img, cmap='gray')
            axes[0, 0].set_title('Imagen Original')
            axes[0, 0].axis('off')
            
            # Mapa LBP
            im1 = axes[0, 1].imshow(lbp, cmap='hot')
            axes[0, 1].set_title(f'Mapa LBP (R={radius}, P={n_points})')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Histograma LBP
            n_bins = int(lbp.max() + 1)
            axes[1, 0].hist(lbp.ravel(), bins=n_bins, density=True, alpha=0.7)
            axes[1, 0].set_title('Histograma LBP')
            axes[1, 0].set_xlabel('Valor LBP')
            axes[1, 0].set_ylabel('Frecuencia Normalizada')
            
            # Mapa de uniformidad
            uniform_lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
            im2 = axes[1, 1].imshow(uniform_lbp, cmap='viridis')
            axes[1, 1].set_title('LBP Uniforme')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Guardar si se especifica ruta
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualización LBP guardada en: {save_path}")
            
            plt.close()
            
            return lbp
            
        except Exception as e:
            logger.error(f"Error visualizando LBP: {str(e)}")
            raise FeatureVisualizationError(f"Error en visualización LBP: {str(e)}")
    
    def visualize_gabor_filters(self, 
                              image: np.ndarray, 
                              frequencies: List[float] = None,
                              angles: List[int] = None,
                              save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Visualiza respuestas de filtros Gabor
        
        Args:
            image: Imagen en escala de grises
            frequencies: Lista de frecuencias a usar
            angles: Lista de ángulos a usar (en grados)
            save_path: Ruta donde guardar la visualización
            
        Returns:
            Diccionario con respuestas de filtros Gabor
        """
        try:
            if frequencies is None:
                frequencies = [0.1, 0.3, 0.5]
            if angles is None:
                angles = [0, 45, 90, 135]
            
            # Asegurar que la imagen esté en escala de grises
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image.copy()
            
            # Calcular respuestas Gabor
            gabor_responses = {}
            
            # Crear figura para visualización
            n_freq = len(frequencies)
            n_angles = len(angles)
            fig, axes = plt.subplots(n_freq + 1, n_angles + 1, figsize=(16, 12))
            
            # Mostrar imagen original
            axes[0, 0].imshow(gray_img, cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            # Ocultar ejes no utilizados en la primera fila
            for j in range(1, n_angles + 1):
                axes[0, j].axis('off')
            
            # Procesar cada combinación de frecuencia y ángulo
            for i, freq in enumerate(frequencies):
                for j, angle in enumerate(angles):
                    # Convertir ángulo a radianes
                    theta = np.deg2rad(angle)
                    
                    # Aplicar filtro Gabor
                    filtered_real, _ = gabor(gray_img, frequency=freq, theta=theta)
                    
                    # Guardar respuesta
                    key = f"freq_{freq}_angle_{angle}"
                    gabor_responses[key] = filtered_real
                    
                    # Visualizar
                    im = axes[i + 1, j].imshow(filtered_real, cmap='RdBu_r')
                    axes[i + 1, j].set_title(f'f={freq}, θ={angle}°')
                    axes[i + 1, j].axis('off')
                    
                    # Añadir colorbar para la primera columna
                    if j == 0:
                        plt.colorbar(im, ax=axes[i + 1, j])
            
            # Ocultar el último eje si no se usa
            if n_angles < len(axes[0]) - 1:
                for i in range(1, n_freq + 1):
                    axes[i, -1].axis('off')
            
            plt.suptitle('Respuestas de Filtros Gabor', fontsize=16)
            plt.tight_layout()
            
            # Guardar si se especifica ruta
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualización Gabor guardada en: {save_path}")
            
            plt.close()
            
            return gabor_responses
            
        except Exception as e:
            logger.error(f"Error visualizando filtros Gabor: {str(e)}")
            raise FeatureVisualizationError(f"Error en visualización Gabor: {str(e)}")
    
    def create_feature_comparison(self, 
                                image: np.ndarray,
                                sift_keypoints: List[cv2.KeyPoint] = None,
                                orb_keypoints: List[cv2.KeyPoint] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Crea una comparación visual entre diferentes tipos de características
        
        Args:
            image: Imagen original
            sift_keypoints: Puntos clave SIFT
            orb_keypoints: Puntos clave ORB
            save_path: Ruta donde guardar la comparación
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Imagen original
            if len(image.shape) == 2:
                axes[0, 0].imshow(image, cmap='gray')
            else:
                axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Imagen Original')
            axes[0, 0].axis('off')
            
            # SIFT keypoints
            if sift_keypoints:
                img_sift = self._draw_keypoints_simple(image, sift_keypoints, (0, 255, 0))
                axes[0, 1].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
                axes[0, 1].set_title(f'SIFT ({len(sift_keypoints)} puntos)')
            else:
                axes[0, 1].text(0.5, 0.5, 'No hay puntos SIFT', ha='center', va='center')
                axes[0, 1].set_title('SIFT (No disponible)')
            axes[0, 1].axis('off')
            
            # ORB keypoints
            if orb_keypoints:
                img_orb = self._draw_keypoints_simple(image, orb_keypoints, (255, 0, 0))
                axes[0, 2].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
                axes[0, 2].set_title(f'ORB ({len(orb_keypoints)} puntos)')
            else:
                axes[0, 2].text(0.5, 0.5, 'No hay puntos ORB', ha='center', va='center')
                axes[0, 2].set_title('ORB (No disponible)')
            axes[0, 2].axis('off')
            
            # LBP
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            lbp = local_binary_pattern(gray_img, 24, 3, method='uniform')
            im1 = axes[1, 0].imshow(lbp, cmap='hot')
            axes[1, 0].set_title('Textura LBP')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0])
            
            # Gabor
            gabor_real, _ = gabor(gray_img, frequency=0.3, theta=0)
            im2 = axes[1, 1].imshow(gabor_real, cmap='RdBu_r')
            axes[1, 1].set_title('Filtro Gabor')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])
            
            # Estadísticas
            stats_text = "Estadísticas de Características:\n\n"
            if sift_keypoints:
                stats_text += f"SIFT: {len(sift_keypoints)} puntos\n"
                sizes = [kp.size for kp in sift_keypoints]
                stats_text += f"  Tamaño promedio: {np.mean(sizes):.2f}\n"
            if orb_keypoints:
                stats_text += f"ORB: {len(orb_keypoints)} puntos\n"
                responses = [kp.response for kp in orb_keypoints]
                stats_text += f"  Respuesta promedio: {np.mean(responses):.4f}\n"
            
            stats_text += f"\nLBP:\n  Valores únicos: {len(np.unique(lbp))}\n"
            stats_text += f"  Uniformidad: {np.sum(lbp <= 24) / lbp.size:.3f}\n"
            
            axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Estadísticas')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Guardar si se especifica ruta
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Comparación de características guardada en: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creando comparación de características: {str(e)}")
            raise FeatureVisualizationError(f"Error en comparación: {str(e)}")
    
    def _draw_keypoints_simple(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], color: Tuple[int, int, int]) -> np.ndarray:
        """
        Dibuja puntos clave de forma simple sobre la imagen
        
        Args:
            image: Imagen base
            keypoints: Lista de puntos clave
            color: Color para dibujar (BGR)
            
        Returns:
            Imagen con puntos clave dibujados
        """
        if image is None:
            raise FeatureVisualizationError("La imagen no puede ser None")
            
        if len(image.shape) == 2:
            img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()
        
        # Si no hay keypoints, devolver la imagen original
        if not keypoints:
            return img_color
        
        return cv2.drawKeypoints(img_color, keypoints, None, color=color, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    def generate_comprehensive_report(self, 
                                    image: np.ndarray,
                                    keypoints: List[cv2.KeyPoint] = None,
                                    descriptors: np.ndarray = None,
                                    output_prefix: str = "features") -> Dict[str, str]:
        """
        Genera un reporte completo con todas las visualizaciones de características
        
        Args:
            image: Imagen a analizar
            keypoints: Lista de puntos clave detectados
            descriptors: Descriptores de características
            output_prefix: Prefijo para los archivos de salida
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        generated_files = {}
        
        try:
            # Validar que la imagen no sea None
            if image is None:
                logger.error("La imagen proporcionada es None en generate_comprehensive_report")
                raise FeatureVisualizationError("No se puede generar reporte con imagen None")
            
            # Validar que la imagen sea un array de numpy válido
            if not isinstance(image, np.ndarray):
                logger.error(f"La imagen debe ser un array de numpy, recibido: {type(image)}")
                raise FeatureVisualizationError(f"Tipo de imagen inválido: {type(image)}")
            
            # Visualización de keypoints si están disponibles
            if keypoints:
                keypoints_path = self.output_dir / f"{output_prefix}_keypoints.png"
                self.visualize_keypoints(image, keypoints, "SIFT", str(keypoints_path))
                generated_files['keypoints'] = str(keypoints_path)
            
            # Visualización de textura LBP
            lbp_path = self.output_dir / f"{output_prefix}_lbp.png"
            self.visualize_lbp_texture(image, save_path=str(lbp_path))
            generated_files['lbp_texture'] = str(lbp_path)
            
            # Visualización de filtros Gabor
            gabor_path = self.output_dir / f"{output_prefix}_gabor.png"
            gabor_results = self.visualize_gabor_filters(image, save_path=str(gabor_path))
            generated_files['gabor_filters'] = str(gabor_path)
            
            logger.info(f"Reporte completo de características generado: {len(generated_files)} archivos")
            
        except Exception as e:
            logger.error(f"Error generando reporte completo: {e}")
            raise FeatureVisualizationError(f"Error en reporte completo: {e}")
            
        return generated_files

    def generate_feature_report(self, 
                              image_path: str,
                              features_data: Dict[str, Any],
                              save_dir: Optional[str] = None) -> str:
        """
        Genera un reporte completo de visualización de características
        
        Args:
            image_path: Ruta de la imagen analizada
            features_data: Datos de características extraídas
            save_dir: Directorio donde guardar el reporte
            
        Returns:
            Ruta del reporte generado
        """
        try:
            if save_dir is None:
                save_dir = self.output_dir / "reports"
            else:
                save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise FeatureVisualizationError(f"No se pudo cargar la imagen: {image_path}")
            
            base_name = Path(image_path).stem
            
            # Generar visualizaciones individuales
            visualizations = {}
            
            # Keypoints si están disponibles
            if 'sift_keypoints' in features_data:
                sift_path = save_dir / f"{base_name}_sift_keypoints.png"
                self.visualize_keypoints(image, features_data['sift_keypoints'], 
                                       "SIFT", str(sift_path))
                visualizations['sift'] = str(sift_path)
            
            if 'orb_keypoints' in features_data:
                orb_path = save_dir / f"{base_name}_orb_keypoints.png"
                self.visualize_keypoints(image, features_data['orb_keypoints'], 
                                       "ORB", str(orb_path))
                visualizations['orb'] = str(orb_path)
            
            # LBP
            lbp_path = save_dir / f"{base_name}_lbp_texture.png"
            self.visualize_lbp_texture(image, save_path=str(lbp_path))
            visualizations['lbp'] = str(lbp_path)
            
            # Gabor
            gabor_path = save_dir / f"{base_name}_gabor_filters.png"
            self.visualize_gabor_filters(image, save_path=str(gabor_path))
            visualizations['gabor'] = str(gabor_path)
            
            # Comparación general
            comparison_path = save_dir / f"{base_name}_feature_comparison.png"
            self.create_feature_comparison(
                image,
                features_data.get('sift_keypoints'),
                features_data.get('orb_keypoints'),
                str(comparison_path)
            )
            visualizations['comparison'] = str(comparison_path)
            
            logger.info(f"Reporte de características generado en: {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            logger.error(f"Error generando reporte de características: {str(e)}")
            raise FeatureVisualizationError(f"Error en reporte: {str(e)}")

def test_feature_visualizer():
    """
    Función de prueba para el visualizador de características
    """
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    # Añadir algunos patrones para hacer más interesante
    cv2.circle(test_image, (100, 100), 50, (255, 255, 255), -1)
    cv2.rectangle(test_image, (200, 200), (300, 300), (128, 128, 128), -1)
    
    # Crear visualizador
    visualizer = FeatureVisualizer("test_visualizations")
    
    # Detectar características SIFT
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints_sift, _ = sift.detectAndCompute(gray, None)
    
    # Detectar características ORB
    orb = cv2.ORB_create()
    keypoints_orb, _ = orb.detectAndCompute(gray, None)
    
    # Generar visualizaciones
    visualizer.visualize_keypoints(test_image, keypoints_sift, "SIFT", "test_sift.png")
    visualizer.visualize_keypoints(test_image, keypoints_orb, "ORB", "test_orb.png")
    visualizer.visualize_lbp_texture(test_image, save_path="test_lbp.png")
    visualizer.visualize_gabor_filters(test_image, save_path="test_gabor.png")
    visualizer.create_feature_comparison(test_image, keypoints_sift, keypoints_orb, "test_comparison.png")
    
    print("Visualizaciones de prueba generadas exitosamente")

if __name__ == "__main__":
    test_feature_visualizer()