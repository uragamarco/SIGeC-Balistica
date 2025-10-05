"""
Visualizador de Preprocesamiento de Imágenes Balísticas
Sistema Balístico Forense MVP

Módulo para visualizar los pasos intermedios del preprocesamiento de imágenes balísticas.
Permite guardar y mostrar las transformaciones aplicadas en cada etapa del proceso.

Funcionalidades:
- Visualización de pasos intermedios del preprocesamiento
- Comparación antes/después de cada transformación
- Generación de mosaicos de visualización
- Guardado de imágenes intermedias
- Métricas de calidad por paso
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import os
from datetime import datetime

from .unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig, PreprocessingResult
from utils.logger import LoggerMixin

@dataclass
class ProcessingStep:
    """Información de un paso de procesamiento"""
    name: str
    description: str
    image_before: np.ndarray
    image_after: np.ndarray
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0

@dataclass
class VisualizationResult:
    """Resultado de la visualización del preprocesamiento"""
    steps: List[ProcessingStep]
    original_image: np.ndarray
    final_image: np.ndarray
    visualization_path: Optional[str] = None
    individual_steps_dir: Optional[str] = None
    success: bool = True
    error_message: str = ""

class PreprocessingVisualizer(LoggerMixin):
    """Visualizador de pasos de preprocesamiento"""
    
    def __init__(self, preprocessor: Optional[UnifiedPreprocessor] = None):
        """
        Inicializa el visualizador
        
        Args:
            preprocessor: Instancia del preprocesador unificado (opcional)
        """
        super().__init__()
        self.preprocessor = preprocessor or UnifiedPreprocessor()
        self.steps = []
        
        # Configuración de visualización
        self.figure_size = (20, 12)
        self.dpi = 150
        self.save_individual_steps = True
        
        self.logger.info("Visualizador de preprocesamiento inicializado")
    
    def preprocess_with_visualization(self, 
                                    image_path: str,
                                    evidence_type: str = "unknown",
                                    level: Optional[str] = None,
                                    output_dir: Optional[str] = None,
                                    save_steps: bool = True) -> VisualizationResult:
        """
        Ejecuta el preprocesamiento con visualización de pasos intermedios
        
        Args:
            image_path: Ruta de la imagen
            evidence_type: Tipo de evidencia
            level: Nivel de preprocesamiento
            output_dir: Directorio para guardar visualizaciones
            save_steps: Si guardar pasos individuales
            
        Returns:
            Resultado de la visualización
        """
        try:
            # Limpiar pasos anteriores
            self.steps = []
            
            # Cargar imagen original
            original_image = self.preprocessor.load_image(image_path)
            if original_image is None:
                return VisualizationResult(
                    steps=[],
                    original_image=None,
                    final_image=None,
                    success=False,
                    error_message="No se pudo cargar la imagen"
                )
            
            # Configurar directorio de salida
            if output_dir is None:
                output_dir = f"preprocessing_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            steps_dir = output_path / "individual_steps" if save_steps else None
            if steps_dir:
                steps_dir.mkdir(exist_ok=True)
            
            # Ejecutar preprocesamiento con captura de pasos
            final_image = self._preprocess_with_step_capture(
                original_image, image_path, evidence_type, level, steps_dir
            )
            
            # Crear visualización completa
            visualization_path = None
            if len(self.steps) > 0:
                visualization_path = str(output_path / "preprocessing_visualization.png")
                self._create_complete_visualization(visualization_path)
            
            return VisualizationResult(
                steps=self.steps.copy(),
                original_image=original_image,
                final_image=final_image,
                visualization_path=visualization_path,
                individual_steps_dir=str(steps_dir) if steps_dir else None,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error en visualización de preprocesamiento: {e}")
            return VisualizationResult(
                steps=[],
                original_image=None,
                final_image=None,
                success=False,
                error_message=str(e)
            )
    
    def _preprocess_with_step_capture(self, 
                                    original_image: np.ndarray,
                                    image_path: str,
                                    evidence_type: str,
                                    level: Optional[str],
                                    steps_dir: Optional[Path]) -> np.ndarray:
        """
        Ejecuta el preprocesamiento capturando cada paso
        
        Args:
            original_image: Imagen original
            image_path: Ruta de la imagen
            evidence_type: Tipo de evidencia
            level: Nivel de preprocesamiento
            steps_dir: Directorio para guardar pasos individuales
            
        Returns:
            Imagen final procesada
        """
        import time
        
        # Configurar nivel si se especifica
        config = self.preprocessor.config
        if level is not None:
            from .unified_preprocessor import PreprocessingLevel
            level_enum = next((l for l in PreprocessingLevel if l.value == level.lower()), None)
            if level_enum and level_enum in self.preprocessor.default_configs:
                config = self.preprocessor.default_configs[level_enum]
        
        # Iniciar con imagen original
        current_image = original_image.copy()
        step_counter = 0
        
        # Paso 0: Imagen original
        self._add_step(
            "original", "Imagen Original", 
            original_image, current_image, 
            {}, steps_dir, step_counter
        )
        step_counter += 1
        
        # Convertir a escala de grises si es necesario
        if len(current_image.shape) == 3 and evidence_type != "unknown":
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.convert_to_grayscale(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "grayscale", "Conversión a Escala de Grises",
                before_image, current_image,
                {"channels": "RGB -> Grayscale"},
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 1: Corrección de iluminación
        if config.illumination_correction:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.correct_illumination(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "illumination", "Corrección de Iluminación",
                before_image, current_image,
                {
                    "clahe_clip_limit": config.clahe_clip_limit,
                    "clahe_tile_size": config.clahe_tile_size
                },
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 2: Reducción de ruido
        if config.noise_reduction:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.reduce_noise(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "noise_reduction", "Reducción de Ruido",
                before_image, current_image,
                {
                    "bilateral_d": config.bilateral_d,
                    "sigma_color": config.bilateral_sigma_color,
                    "sigma_space": config.bilateral_sigma_space
                },
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 3: Mejora de contraste
        if config.contrast_enhancement:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.enhance_contrast(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "contrast", "Mejora de Contraste",
                before_image, current_image,
                {
                    "gamma_correction": config.gamma_correction,
                    "histogram_equalization": config.histogram_equalization
                },
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 4: Detección y corrección de rotación
        if config.normalize_orientation:
            before_image = current_image.copy()
            start_time = time.time()
            current_image, rotation_angle = self.preprocessor.detect_and_correct_rotation(current_image)
            processing_time = time.time() - start_time
            
            if abs(rotation_angle) > 1.0:
                self._add_step(
                    "rotation", "Corrección de Rotación",
                    before_image, current_image,
                    {"rotation_angle": f"{rotation_angle:.2f}°"},
                    steps_dir, step_counter, processing_time
                )
                step_counter += 1
        
        # Paso 5: Mejora de bordes
        if config.edge_enhancement:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.enhance_edges(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "edges", "Mejora de Bordes",
                before_image, current_image,
                {"method": "Unsharp mask + Laplacian"},
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 6: Operaciones morfológicas
        if config.morphological_operations:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.apply_morphological_operations(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "morphology", "Operaciones Morfológicas",
                before_image, current_image,
                {"operations": "Opening + Closing"},
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 7: Redimensionamiento
        if config.resize_images:
            before_image = current_image.copy()
            original_size = (current_image.shape[1], current_image.shape[0])
            start_time = time.time()
            current_image = self.preprocessor.resize_image(current_image, config.target_size)
            processing_time = time.time() - start_time
            
            self._add_step(
                "resize", "Redimensionamiento",
                before_image, current_image,
                {
                    "from_size": f"{original_size[0]}x{original_size[1]}",
                    "to_size": f"{config.target_size[0]}x{config.target_size[1]}"
                },
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        # Paso 8: Mejoras específicas según tipo de evidencia
        if evidence_type == "bullet" and config.enhance_striations:
            before_image = current_image.copy()
            start_time = time.time()
            current_image = self.preprocessor.enhance_striations(current_image)
            processing_time = time.time() - start_time
            
            self._add_step(
                "striations", "Mejora de Estrías",
                before_image, current_image,
                {"evidence_type": "bullet"},
                steps_dir, step_counter, processing_time
            )
            step_counter += 1
        
        if evidence_type == "cartridge_case":
            if config.enhance_breech_marks:
                before_image = current_image.copy()
                start_time = time.time()
                current_image = self.preprocessor.enhance_breech_marks(current_image)
                processing_time = time.time() - start_time
                
                self._add_step(
                    "breech_marks", "Mejora de Marcas de Recámara",
                    before_image, current_image,
                    {"evidence_type": "cartridge_case"},
                    steps_dir, step_counter, processing_time
                )
                step_counter += 1
            
            if config.enhance_firing_pin:
                before_image = current_image.copy()
                start_time = time.time()
                current_image = self.preprocessor.enhance_firing_pin(current_image)
                processing_time = time.time() - start_time
                
                self._add_step(
                    "firing_pin", "Mejora de Marca de Percutor",
                    before_image, current_image,
                    {"evidence_type": "cartridge_case"},
                    steps_dir, step_counter, processing_time
                )
                step_counter += 1
        
        return current_image
    
    def _add_step(self, 
                 name: str, 
                 description: str,
                 image_before: np.ndarray,
                 image_after: np.ndarray,
                 parameters: Dict[str, Any],
                 steps_dir: Optional[Path],
                 step_number: int,
                 processing_time: float = 0.0):
        """
        Añade un paso de procesamiento a la lista
        
        Args:
            name: Nombre del paso
            description: Descripción del paso
            image_before: Imagen antes del procesamiento
            image_after: Imagen después del procesamiento
            parameters: Parámetros utilizados
            steps_dir: Directorio para guardar pasos individuales
            step_number: Número del paso
            processing_time: Tiempo de procesamiento
        """
        # Calcular métricas básicas
        metrics = self._calculate_step_metrics(image_before, image_after)
        
        # Crear paso
        step = ProcessingStep(
            name=name,
            description=description,
            image_before=image_before.copy(),
            image_after=image_after.copy(),
            parameters=parameters,
            metrics=metrics,
            processing_time=processing_time
        )
        
        self.steps.append(step)
        
        # Guardar paso individual si se solicita
        if steps_dir and self.save_individual_steps:
            self._save_individual_step(step, steps_dir, step_number)
    
    def _calculate_step_metrics(self, 
                              image_before: np.ndarray, 
                              image_after: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas básicas para un paso de procesamiento
        
        Args:
            image_before: Imagen antes del procesamiento
            image_after: Imagen después del procesamiento
            
        Returns:
            Diccionario con métricas
        """
        try:
            metrics = {}
            
            # Asegurar que las imágenes tengan el mismo tamaño para comparación
            if image_before.shape != image_after.shape:
                # Si tienen diferentes tamaños, redimensionar la imagen before
                image_before_resized = cv2.resize(image_before, 
                                                (image_after.shape[1], image_after.shape[0]))
            else:
                image_before_resized = image_before
            
            # Convertir a escala de grises si es necesario
            if len(image_before_resized.shape) == 3:
                gray_before = cv2.cvtColor(image_before_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray_before = image_before_resized
                
            if len(image_after.shape) == 3:
                gray_after = cv2.cvtColor(image_after, cv2.COLOR_BGR2GRAY)
            else:
                gray_after = image_after
            
            # Contraste (desviación estándar)
            contrast_before = np.std(gray_before)
            contrast_after = np.std(gray_after)
            metrics['contrast_before'] = float(contrast_before)
            metrics['contrast_after'] = float(contrast_after)
            metrics['contrast_improvement'] = float(contrast_after - contrast_before)
            
            # Nitidez (varianza del Laplaciano)
            laplacian_before = cv2.Laplacian(gray_before, cv2.CV_64F)
            laplacian_after = cv2.Laplacian(gray_after, cv2.CV_64F)
            sharpness_before = np.var(laplacian_before)
            sharpness_after = np.var(laplacian_after)
            metrics['sharpness_before'] = float(sharpness_before)
            metrics['sharpness_after'] = float(sharpness_after)
            metrics['sharpness_improvement'] = float(sharpness_after - sharpness_before)
            
            # Entropía (medida de información)
            hist_before = cv2.calcHist([gray_before], [0], None, [256], [0, 256])
            hist_after = cv2.calcHist([gray_after], [0], None, [256], [0, 256])
            
            # Normalizar histogramas
            hist_before = hist_before / hist_before.sum()
            hist_after = hist_after / hist_after.sum()
            
            # Calcular entropía
            entropy_before = -np.sum(hist_before * np.log2(hist_before + 1e-10))
            entropy_after = -np.sum(hist_after * np.log2(hist_after + 1e-10))
            metrics['entropy_before'] = float(entropy_before)
            metrics['entropy_after'] = float(entropy_after)
            metrics['entropy_change'] = float(entropy_after - entropy_before)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculando métricas del paso: {e}")
            return {}
    
    def _save_individual_step(self, 
                            step: ProcessingStep, 
                            steps_dir: Path, 
                            step_number: int):
        """
        Guarda la visualización de un paso individual
        
        Args:
            step: Paso de procesamiento
            steps_dir: Directorio de pasos
            step_number: Número del paso
        """
        try:
            # Crear figura para el paso individual
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f"Paso {step_number}: {step.description}", fontsize=14, fontweight='bold')
            
            # Imagen antes
            if len(step.image_before.shape) == 3:
                axes[0].imshow(cv2.cvtColor(step.image_before, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(step.image_before, cmap='gray')
            axes[0].set_title("Antes")
            axes[0].axis('off')
            
            # Imagen después
            if len(step.image_after.shape) == 3:
                axes[1].imshow(cv2.cvtColor(step.image_after, cv2.COLOR_BGR2RGB))
            else:
                axes[1].imshow(step.image_after, cmap='gray')
            axes[1].set_title("Después")
            axes[1].axis('off')
            
            # Añadir información del paso
            info_text = []
            if step.parameters:
                info_text.append("Parámetros:")
                for key, value in step.parameters.items():
                    info_text.append(f"  {key}: {value}")
            
            if step.metrics:
                info_text.append("\nMétricas:")
                for key, value in step.metrics.items():
                    if isinstance(value, float):
                        info_text.append(f"  {key}: {value:.3f}")
                    else:
                        info_text.append(f"  {key}: {value}")
            
            if step.processing_time > 0:
                info_text.append(f"\nTiempo: {step.processing_time:.3f}s")
            
            if info_text:
                plt.figtext(0.02, 0.02, '\n'.join(info_text), fontsize=8, 
                           verticalalignment='bottom', fontfamily='monospace')
            
            # Guardar
            step_filename = f"step_{step_number:02d}_{step.name}.png"
            step_path = steps_dir / step_filename
            plt.savefig(step_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"Paso individual guardado: {step_path}")
            
        except Exception as e:
            self.logger.error(f"Error guardando paso individual: {e}")
    
    def _create_complete_visualization(self, output_path: str):
        """
        Crea la visualización completa de todos los pasos
        
        Args:
            output_path: Ruta de salida para la visualización
        """
        try:
            if not self.steps:
                return
            
            # Calcular layout de la grilla
            n_steps = len(self.steps)
            cols = min(4, n_steps)  # Máximo 4 columnas
            rows = (n_steps + cols - 1) // cols  # Redondear hacia arriba
            
            # Crear figura
            fig = plt.figure(figsize=self.figure_size)
            gs = gridspec.GridSpec(rows, cols, figure=fig)
            
            # Título principal
            fig.suptitle('Visualización del Preprocesamiento de Imagen Balística', 
                        fontsize=16, fontweight='bold')
            
            # Mostrar cada paso
            for i, step in enumerate(self.steps):
                row = i // cols
                col = i % cols
                
                ax = fig.add_subplot(gs[row, col])
                
                # Mostrar imagen después del procesamiento
                if len(step.image_after.shape) == 3:
                    ax.imshow(cv2.cvtColor(step.image_after, cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(step.image_after, cmap='gray')
                
                ax.set_title(f"{i}: {step.description}", fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Añadir información básica
                info_lines = []
                if step.processing_time > 0:
                    info_lines.append(f"Tiempo: {step.processing_time:.3f}s")
                
                # Mostrar mejora de contraste si está disponible
                if 'contrast_improvement' in step.metrics:
                    improvement = step.metrics['contrast_improvement']
                    if improvement > 0:
                        info_lines.append(f"↑Contraste: +{improvement:.1f}")
                    elif improvement < 0:
                        info_lines.append(f"↓Contraste: {improvement:.1f}")
                
                if info_lines:
                    ax.text(0.02, 0.02, '\n'.join(info_lines), 
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Ajustar layout y guardar
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualización completa guardada: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creando visualización completa: {e}")
    
    def create_comparison_visualization(self, 
                                     original_image: np.ndarray,
                                     final_image: np.ndarray,
                                     output_path: str):
        """
        Crea una visualización de comparación antes/después
        
        Args:
            original_image: Imagen original
            final_image: Imagen final procesada
            output_path: Ruta de salida
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            fig.suptitle('Comparación: Antes vs Después del Preprocesamiento', 
                        fontsize=16, fontweight='bold')
            
            # Imagen original
            if len(original_image.shape) == 3:
                axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(original_image, cmap='gray')
            axes[0].set_title("Imagen Original", fontsize=14)
            axes[0].axis('off')
            
            # Imagen procesada
            if len(final_image.shape) == 3:
                axes[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            else:
                axes[1].imshow(final_image, cmap='gray')
            axes[1].set_title("Imagen Procesada", fontsize=14)
            axes[1].axis('off')
            
            # Calcular y mostrar métricas de mejora
            metrics = self._calculate_step_metrics(original_image, final_image)
            
            info_text = []
            if 'contrast_improvement' in metrics:
                info_text.append(f"Mejora de contraste: {metrics['contrast_improvement']:.3f}")
            if 'sharpness_improvement' in metrics:
                info_text.append(f"Mejora de nitidez: {metrics['sharpness_improvement']:.3f}")
            if 'entropy_change' in metrics:
                info_text.append(f"Cambio de entropía: {metrics['entropy_change']:.3f}")
            
            if info_text:
                plt.figtext(0.5, 0.02, ' | '.join(info_text), 
                           ha='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Visualización de comparación guardada: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creando visualización de comparación: {e}")


# Función de ayuda para uso directo
def visualize_preprocessing(image_path: str,
                          evidence_type: str = "unknown",
                          level: str = "standard",
                          output_dir: Optional[str] = None) -> VisualizationResult:
    """
    Función de ayuda para visualizar el preprocesamiento de una imagen
    
    Args:
        image_path: Ruta de la imagen
        evidence_type: Tipo de evidencia
        level: Nivel de preprocesamiento
        output_dir: Directorio de salida
        
    Returns:
        Resultado de la visualización
    """
    visualizer = PreprocessingVisualizer()
    return visualizer.preprocess_with_visualization(
        image_path, evidence_type, level, output_dir
    )


if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear parser de argumentos
    parser = argparse.ArgumentParser(description="Visualizador de preprocesamiento de imágenes balísticas")
    parser.add_argument("input", help="Imagen de entrada")
    parser.add_argument("--output", "-o", help="Directorio de salida", default="preprocessing_viz")
    parser.add_argument("--level", "-l", help="Nivel de preprocesamiento", 
                       choices=["basic", "standard", "advanced", "forensic"], default="standard")
    parser.add_argument("--evidence", "-e", help="Tipo de evidencia", 
                       choices=["cartridge_case", "bullet", "unknown"], default="unknown")
    
    args = parser.parse_args()
    
    # Ejecutar visualización
    result = visualize_preprocessing(
        args.input,
        evidence_type=args.evidence,
        level=args.level,
        output_dir=args.output
    )
    
    if result.success:
        print(f"\n=== VISUALIZACIÓN COMPLETADA ===")
        print(f"Imagen: {Path(args.input).name}")
        print(f"Pasos procesados: {len(result.steps)}")
        print(f"Visualización: {result.visualization_path}")
        if result.individual_steps_dir:
            print(f"Pasos individuales: {result.individual_steps_dir}")
    else:
        print(f"Error: {result.error_message}")