#!/usr/bin/env python3
"""
Visualizador de ROI (Regiones de Interés)
Genera visualizaciones de las regiones detectadas automáticamente
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo antes de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime

from utils.logger import LoggerMixin


class ROIVisualizer(LoggerMixin):
    """
    Visualizador especializado para ROI (Regiones de Interés)
    """
    
    def __init__(self, output_dir: str = "temp/visualizations"):
        """
        Inicializa el visualizador de ROI
        
        Args:
            output_dir: Directorio de salida para las visualizaciones
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de colores para diferentes tipos de ROI
        self.roi_colors = {
            'cartridge_case': '#FF6B6B',  # Rojo para vainas
            'bullet': '#4ECDC4',          # Turquesa para proyectiles
            'enhanced_watershed': '#45B7D1',  # Azul para Watershed mejorado
            'circle_detection': '#96CEB4',    # Verde para detección circular
            'contour_detection': '#FFEAA7',   # Amarillo para contornos
            'edge_detection': '#DDA0DD',      # Púrpura para bordes
            'default': '#74B9FF'              # Azul por defecto
        }
        
        # Configuración de matplotlib
        plt.style.use('default')
        
    def generate_comprehensive_report(self, 
                                    image_path: str,
                                    roi_regions: List[Dict[str, Any]],
                                    evidence_type: str = "unknown",
                                    output_prefix: str = "roi") -> Dict[str, str]:
        """
        Genera un reporte completo de visualización de ROI
        
        Args:
            image_path: Ruta de la imagen original
            roi_regions: Lista de regiones ROI detectadas
            evidence_type: Tipo de evidencia (cartridge_case, bullet)
            output_prefix: Prefijo para archivos de salida
            
        Returns:
            Diccionario con rutas de visualizaciones generadas
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            visualizations = {}
            
            # 1. Visualización general de todas las ROI
            overview_path = self._generate_roi_overview(
                image_rgb, roi_regions, evidence_type, f"{output_prefix}_overview"
            )
            visualizations['overview'] = str(overview_path)
            
            # 2. Visualización detallada por tipo de detección
            detailed_path = self._generate_detailed_roi_view(
                image_rgb, roi_regions, evidence_type, f"{output_prefix}_detailed"
            )
            visualizations['detailed'] = str(detailed_path)
            
            # 3. Visualización de estadísticas de ROI
            stats_path = self._generate_roi_statistics(
                roi_regions, evidence_type, f"{output_prefix}_statistics"
            )
            visualizations['statistics'] = str(stats_path)
            
            # 4. Visualización individual de cada ROI (si hay pocas)
            if len(roi_regions) <= 10:
                individual_path = self._generate_individual_roi_views(
                    image_rgb, roi_regions, evidence_type, f"{output_prefix}_individual"
                )
                visualizations['individual'] = str(individual_path)
            
            # 5. Mapa de calor de confianza
            heatmap_path = self._generate_confidence_heatmap(
                image_rgb, roi_regions, f"{output_prefix}_heatmap"
            )
            visualizations['heatmap'] = str(heatmap_path)
            
            self.logger.info(f"Generadas {len(visualizations)} visualizaciones de ROI")
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de ROI: {str(e)}")
            raise
    
    def _generate_roi_overview(self, 
                              image: np.ndarray,
                              roi_regions: List[Dict[str, Any]],
                              evidence_type: str,
                              output_name: str) -> Path:
        """
        Genera visualización general de todas las ROI
        """
        # Configurar matplotlib para usar backend no interactivo
        plt.switch_backend('Agg')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Verificar que ax es un objeto Axes válido
        if not hasattr(ax, 'imshow'):
            self.logger.error("Error: ax no es un objeto matplotlib Axes válido")
            raise ValueError("Error en la creación de la figura matplotlib")
        
        # Mostrar imagen original
        ax.imshow(image)
        ax.set_title(f'Detección de ROI - {evidence_type.replace("_", " ").title()}\n'
                    f'Total: {len(roi_regions)} regiones detectadas', 
                    fontsize=14, fontweight='bold')
        
        # Dibujar cada ROI
        for i, roi in enumerate(roi_regions):
            self._draw_roi_region(ax, roi, i)
        
        # Leyenda
        self._add_roi_legend(ax, roi_regions)
        
        ax.axis('off')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _generate_detailed_roi_view(self,
                                   image: np.ndarray,
                                   roi_regions: List[Dict[str, Any]],
                                   evidence_type: str,
                                   output_name: str) -> Path:
        """
        Genera visualización detallada por tipo de detección
        """
        # Agrupar ROI por método de detección
        roi_by_method = {}
        for roi in roi_regions:
            method = roi.get('detection_method', 'unknown')
            if method not in roi_by_method:
                roi_by_method[method] = []
            roi_by_method[method].append(roi)
        
        n_methods = len(roi_by_method)
        if n_methods == 0:
            # Crear imagen vacía si no hay ROI
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No se detectaron ROI', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
        else:
            # Crear subplots para cada método
            cols = min(2, n_methods)
            rows = (n_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
            if n_methods == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, (method, rois) in enumerate(roi_by_method.items()):
                ax = axes[idx] if n_methods > 1 else axes[0]
                
                # Mostrar imagen
                ax.imshow(image)
                ax.set_title(f'{method.replace("_", " ").title()}\n'
                           f'{len(rois)} regiones', fontweight='bold')
                
                # Dibujar ROI de este método
                for i, roi in enumerate(rois):
                    self._draw_roi_region(ax, roi, i, method)
                
                ax.axis('off')
            
            # Ocultar subplots vacíos
            for idx in range(n_methods, len(axes)):
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _generate_roi_statistics(self,
                                roi_regions: List[Dict[str, Any]],
                                evidence_type: str,
                                output_name: str) -> Path:
        """
        Genera visualización de estadísticas de ROI
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        if not roi_regions:
            # Si no hay ROI, mostrar mensaje
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No hay datos de ROI', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            # 1. Distribución por método de detección
            methods = [roi.get('detection_method', 'unknown') for roi in roi_regions]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            ax1.pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
            ax1.set_title('Distribución por Método de Detección')
            
            # 2. Distribución de confianza
            confidences = [roi.get('confidence', 0.0) for roi in roi_regions]
            ax2.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Confianza')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Distribución de Confianza')
            ax2.grid(True, alpha=0.3)
            
            # 3. Distribución de tamaños de ROI
            areas = []
            for roi in roi_regions:
                if 'bbox' in roi:
                    bbox = roi['bbox']
                    area = bbox[2] * bbox[3]  # width * height
                    areas.append(area)
            
            if areas:
                ax3.hist(areas, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_xlabel('Área (píxeles²)')
                ax3.set_ylabel('Frecuencia')
                ax3.set_title('Distribución de Tamaños de ROI')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No hay datos de área', ha='center', va='center')
                ax3.axis('off')
            
            # 4. Estadísticas textuales
            ax4.axis('off')
            stats_text = self._generate_roi_stats_text(roi_regions)
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _generate_individual_roi_views(self,
                                      image: np.ndarray,
                                      roi_regions: List[Dict[str, Any]],
                                      evidence_type: str,
                                      output_name: str) -> Path:
        """
        Genera visualización individual de cada ROI
        """
        n_rois = len(roi_regions)
        if n_rois == 0:
            # Crear imagen vacía
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No hay ROI para mostrar', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
        else:
            cols = min(3, n_rois)
            rows = (n_rois + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if n_rois == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, roi in enumerate(roi_regions):
                ax = axes[i] if n_rois > 1 else axes[0]
                
                # Extraer región de la imagen
                roi_image = self._extract_roi_region(image, roi)
                
                if roi_image is not None:
                    ax.imshow(roi_image)
                    method = roi.get('detection_method', 'unknown')
                    confidence = roi.get('confidence', 0.0)
                    ax.set_title(f'ROI {i+1}\n{method}\nConf: {confidence:.3f}', 
                               fontsize=10)
                else:
                    ax.text(0.5, 0.5, f'ROI {i+1}\nError', 
                           ha='center', va='center')
                
                ax.axis('off')
            
            # Ocultar subplots vacíos
            for i in range(n_rois, len(axes)):
                axes[i].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _generate_confidence_heatmap(self,
                                   image: np.ndarray,
                                   roi_regions: List[Dict[str, Any]],
                                   output_name: str) -> Path:
        """
        Genera mapa de calor de confianza de las ROI
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Imagen original
        ax1.imshow(image)
        ax1.set_title('Imagen Original', fontweight='bold')
        ax1.axis('off')
        
        # Mapa de calor
        if roi_regions:
            # Crear mapa de confianza
            h, w = image.shape[:2]
            confidence_map = np.zeros((h, w))
            
            for roi in roi_regions:
                confidence = roi.get('confidence', 0.0)
                
                if 'bbox' in roi:
                    x, y, width, height = roi['bbox']
                    x, y = int(x), int(y)
                    width, height = int(width), int(height)
                    
                    # Asegurar que las coordenadas estén dentro de la imagen
                    x = max(0, min(x, w-1))
                    y = max(0, min(y, h-1))
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    if width > 0 and height > 0:
                        confidence_map[y:y+height, x:x+width] = max(
                            confidence_map[y:y+height, x:x+width].max(), confidence
                        )
            
            # Mostrar mapa de calor
            im = ax2.imshow(confidence_map, cmap='hot', alpha=0.8)
            ax2.imshow(image, alpha=0.3)  # Overlay de la imagen original
            ax2.set_title('Mapa de Calor de Confianza', fontweight='bold')
            
            # Barra de color
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Confianza', rotation=270, labelpad=15)
        else:
            ax2.imshow(image)
            ax2.text(0.5, 0.5, 'No hay datos de confianza', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='white', fontweight='bold')
            ax2.set_title('Mapa de Calor de Confianza', fontweight='bold')
        
        ax2.axis('off')
        plt.tight_layout()
        
        output_path = self.output_dir / f"{output_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _draw_roi_region(self, ax, roi: Dict[str, Any], index: int, method: str = None):
        """
        Dibuja una región ROI en el axes
        """
        # Obtener color según el método
        detection_method = method or roi.get('detection_method', 'default')
        color = self.roi_colors.get(detection_method, self.roi_colors['default'])
        
        # Obtener confianza para el alpha
        confidence = roi.get('confidence', 1.0)
        alpha = max(0.3, min(1.0, confidence))
        
        if 'bbox' in roi:
            # Dibujar rectángulo
            x, y, width, height = roi['bbox']
            rect = Rectangle((x, y), width, height, 
                           linewidth=2, edgecolor=color, 
                           facecolor=color, alpha=alpha*0.3)
            ax.add_patch(rect)
            
            # Etiqueta
            label = f"ROI {index+1}\n{confidence:.3f}"
            ax.text(x, y-5, label, fontsize=8, color=color, 
                   fontweight='bold', ha='left', va='bottom')
        
        elif 'center' in roi and 'radius' in roi:
            # Dibujar círculo
            center = roi['center']
            radius = roi['radius']
            circle = Circle(center, radius, 
                          linewidth=2, edgecolor=color,
                          facecolor=color, alpha=alpha*0.3)
            ax.add_patch(circle)
            
            # Etiqueta
            label = f"ROI {index+1}\n{confidence:.3f}"
            ax.text(center[0], center[1]-radius-10, label, 
                   fontsize=8, color=color, fontweight='bold',
                   ha='center', va='bottom')
        
        elif 'contour' in roi:
            # Dibujar contorno
            contour = np.array(roi['contour'])
            if len(contour) > 2:
                polygon = Polygon(contour, linewidth=2, edgecolor=color,
                                facecolor=color, alpha=alpha*0.3)
                ax.add_patch(polygon)
                
                # Etiqueta en el centroide
                centroid = np.mean(contour, axis=0)
                label = f"ROI {index+1}\n{confidence:.3f}"
                ax.text(centroid[0], centroid[1], label, 
                       fontsize=8, color=color, fontweight='bold',
                       ha='center', va='center')
    
    def _add_roi_legend(self, ax, roi_regions: List[Dict[str, Any]]):
        """
        Añade leyenda a la visualización
        """
        # Obtener métodos únicos
        methods = set()
        for roi in roi_regions:
            methods.add(roi.get('detection_method', 'unknown'))
        
        # Crear elementos de leyenda
        legend_elements = []
        for method in sorted(methods):
            color = self.roi_colors.get(method, self.roi_colors['default'])
            count = sum(1 for roi in roi_regions 
                       if roi.get('detection_method', 'unknown') == method)
            
            legend_elements.append(
                patches.Patch(color=color, 
                            label=f'{method.replace("_", " ").title()} ({count})')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1, 1), fontsize=10)
    
    def _extract_roi_region(self, image: np.ndarray, roi: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extrae la región ROI de la imagen
        """
        try:
            if 'bbox' in roi:
                x, y, width, height = roi['bbox']
                x, y = int(x), int(y)
                width, height = int(width), int(height)
                
                # Asegurar coordenadas válidas
                h, w = image.shape[:2]
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    return image[y:y+height, x:x+width]
            
            elif 'center' in roi and 'radius' in roi:
                center = roi['center']
                radius = int(roi['radius'])
                
                # Crear máscara circular
                h, w = image.shape[:2]
                y, x = np.ogrid[:h, :w]
                mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                
                # Extraer región rectangular que contiene el círculo
                x_min = max(0, int(center[0] - radius))
                x_max = min(w, int(center[0] + radius))
                y_min = max(0, int(center[1] - radius))
                y_max = min(h, int(center[1] + radius))
                
                roi_image = image[y_min:y_max, x_min:x_max].copy()
                roi_mask = mask[y_min:y_max, x_min:x_max]
                
                # Aplicar máscara
                roi_image[~roi_mask] = 0
                return roi_image
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extrayendo región ROI: {str(e)}")
            return None
    
    def _generate_roi_stats_text(self, roi_regions: List[Dict[str, Any]]) -> str:
        """
        Genera texto con estadísticas de ROI
        """
        if not roi_regions:
            return "No hay regiones ROI detectadas"
        
        stats = []
        stats.append("=== ESTADÍSTICAS DE ROI ===\n")
        
        # Estadísticas generales
        stats.append(f"Total de regiones: {len(roi_regions)}")
        
        # Por método de detección
        methods = {}
        confidences = []
        areas = []
        
        for roi in roi_regions:
            method = roi.get('detection_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
            
            confidence = roi.get('confidence', 0.0)
            confidences.append(confidence)
            
            if 'bbox' in roi:
                bbox = roi['bbox']
                area = bbox[2] * bbox[3]
                areas.append(area)
        
        stats.append(f"\nMétodos de detección:")
        for method, count in sorted(methods.items()):
            stats.append(f"  {method}: {count}")
        
        # Estadísticas de confianza
        if confidences:
            stats.append(f"\nConfianza:")
            stats.append(f"  Promedio: {np.mean(confidences):.3f}")
            stats.append(f"  Mínima: {np.min(confidences):.3f}")
            stats.append(f"  Máxima: {np.max(confidences):.3f}")
            stats.append(f"  Desv. Est.: {np.std(confidences):.3f}")
        
        # Estadísticas de área
        if areas:
            stats.append(f"\nÁreas (píxeles²):")
            stats.append(f"  Promedio: {np.mean(areas):.0f}")
            stats.append(f"  Mínima: {np.min(areas):.0f}")
            stats.append(f"  Máxima: {np.max(areas):.0f}")
            stats.append(f"  Desv. Est.: {np.std(areas):.0f}")
        
        return "\n".join(stats)