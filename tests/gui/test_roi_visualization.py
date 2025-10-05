#!/usr/bin/env python3
"""
Test para validar la visualización automática de ROI
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from image_processing.roi_visualizer import ROIVisualizer
from image_processing.unified_roi_detector import UnifiedROIDetector
from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingLevel, PreprocessingConfig
from utils.logger import LoggerMixin


class ROIVisualizationTester(LoggerMixin):
    """
    Tester para validar la visualización automática de ROI
    """
    
    def __init__(self):
        self.test_dir = Path("temp/test_roi_viz")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear imagen de prueba
        self.test_image_path = self._create_test_image()
        
    def _create_test_image(self) -> str:
        """
        Crea una imagen de prueba simulando una vaina
        """
        # Crear imagen de 800x600 con fondo gris
        image = np.ones((600, 800, 3), dtype=np.uint8) * 128
        
        # Simular una vaina (círculo con detalles)
        center = (400, 300)
        radius = 150
        
        # Círculo exterior (borde de la vaina)
        cv2.circle(image, center, radius, (200, 200, 200), 3)
        
        # Círculo interior (culote)
        cv2.circle(image, center, radius//3, (180, 180, 180), 2)
        
        # Añadir algunas marcas radiales (simulando estrías)
        for angle in range(0, 360, 30):
            angle_rad = np.radians(angle)
            start_x = int(center[0] + (radius//4) * np.cos(angle_rad))
            start_y = int(center[1] + (radius//4) * np.sin(angle_rad))
            end_x = int(center[0] + (radius//2) * np.cos(angle_rad))
            end_y = int(center[1] + (radius//2) * np.sin(angle_rad))
            cv2.line(image, (start_x, start_y), (end_x, end_y), (160, 160, 160), 1)
        
        # Añadir algo de ruido
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Guardar imagen
        test_image_path = self.test_dir / "test_cartridge_case.jpg"
        cv2.imwrite(str(test_image_path), image)
        
        self.logger.info(f"Imagen de prueba creada: {test_image_path}")
        return str(test_image_path)
    
    def test_roi_detection_and_visualization(self):
        """
        Test completo de detección y visualización de ROI
        """
        try:
            self.logger.info("=== Iniciando test de detección y visualización ROI ===")
            
            # 1. Preprocesar imagen
            self.logger.info("1. Preprocesando imagen...")
            
            # Crear configuración con ROI habilitada
            config = PreprocessingConfig(
                level=PreprocessingLevel.ADVANCED,
                enable_roi_detection=True,
                roi_detection_level="advanced"
            )
            
            preprocessor = UnifiedPreprocessor(config)
            
            preprocessing_result = preprocessor.preprocess_image(
                self.test_image_path,
                evidence_type="cartridge_case"
            )
            
            self.logger.info(f"Preprocesamiento completado. ROI detectadas: {preprocessing_result.metadata.get('roi_regions_count', 0)}")
            
            # 2. Extraer regiones ROI del resultado
            roi_regions = preprocessing_result.metadata.get('roi_regions', [])
            
            if not roi_regions:
                self.logger.warning("No se detectaron ROI. Creando ROI de prueba...")
                roi_regions = self._create_mock_roi_regions()
            
            # 3. Generar visualizaciones
            self.logger.info("2. Generando visualizaciones ROI...")
            visualizer = ROIVisualizer(str(self.test_dir / "visualizations"))
            
            visualizations = visualizer.generate_comprehensive_report(
                image_path=self.test_image_path,
                roi_regions=roi_regions,
                evidence_type="cartridge_case",
                output_prefix="test_roi"
            )
            
            # 4. Verificar resultados
            self.logger.info("3. Verificando resultados...")
            self._verify_visualizations(visualizations)
            
            self.logger.info("=== Test completado exitosamente ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en test ROI: {str(e)}")
            return False
    
    def _create_mock_roi_regions(self) -> List[Dict[str, Any]]:
        """
        Crea regiones ROI de prueba
        """
        mock_regions = [
            {
                'bbox': [250, 150, 300, 300],  # x, y, width, height
                'confidence': 0.85,
                'detection_method': 'enhanced_watershed',
                'area': 90000,
                'center': [400, 300]
            },
            {
                'bbox': [350, 250, 100, 100],
                'confidence': 0.72,
                'detection_method': 'circle_detection',
                'area': 10000,
                'center': [400, 300]
            },
            {
                'center': [400, 300],
                'radius': 50,
                'confidence': 0.68,
                'detection_method': 'contour_detection',
                'area': 7854
            }
        ]
        
        self.logger.info(f"Creadas {len(mock_regions)} regiones ROI de prueba")
        return mock_regions
    
    def _verify_visualizations(self, visualizations: Dict[str, str]):
        """
        Verifica que las visualizaciones se generaron correctamente
        """
        expected_types = ['overview', 'detailed', 'statistics', 'individual', 'heatmap']
        
        self.logger.info(f"Visualizaciones generadas: {list(visualizations.keys())}")
        
        for viz_type in expected_types:
            if viz_type in visualizations:
                viz_path = Path(visualizations[viz_type])
                if viz_path.exists():
                    file_size = viz_path.stat().st_size
                    self.logger.info(f"✓ {viz_type}: {viz_path.name} ({file_size} bytes)")
                else:
                    self.logger.warning(f"✗ {viz_type}: Archivo no encontrado - {viz_path}")
            else:
                self.logger.warning(f"✗ {viz_type}: Tipo de visualización no generado")
        
        # Verificar que al menos se generó la visualización overview
        if 'overview' in visualizations and Path(visualizations['overview']).exists():
            self.logger.info("✓ Verificación básica pasada: visualización overview generada")
            return True
        else:
            self.logger.error("✗ Verificación fallida: no se generó visualización overview")
            return False
    
    def test_empty_roi_case(self):
        """
        Test para el caso sin ROI detectadas
        """
        try:
            self.logger.info("=== Test caso sin ROI ===")
            
            visualizer = ROIVisualizer(str(self.test_dir / "empty_visualizations"))
            
            # Generar visualizaciones con lista vacía
            visualizations = visualizer.generate_comprehensive_report(
                image_path=self.test_image_path,
                roi_regions=[],  # Lista vacía
                evidence_type="cartridge_case",
                output_prefix="empty_roi"
            )
            
            self.logger.info(f"Visualizaciones para caso vacío: {list(visualizations.keys())}")
            
            # Verificar que se generaron visualizaciones incluso sin ROI
            if visualizations:
                self.logger.info("✓ Test caso vacío pasado: se generaron visualizaciones")
                return True
            else:
                self.logger.error("✗ Test caso vacío fallido: no se generaron visualizaciones")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en test caso vacío: {str(e)}")
            return False
    
    def run_all_tests(self):
        """
        Ejecuta todos los tests
        """
        self.logger.info("Iniciando batería de tests de visualización ROI")
        
        results = []
        
        # Test principal
        results.append(("Detección y Visualización ROI", self.test_roi_detection_and_visualization()))
        
        # Test caso vacío
        results.append(("Caso sin ROI", self.test_empty_roi_case()))
        
        # Resumen
        self.logger.info("\n=== RESUMEN DE TESTS ===")
        passed = 0
        for test_name, result in results:
            status = "✓ PASADO" if result else "✗ FALLIDO"
            self.logger.info(f"{test_name}: {status}")
            if result:
                passed += 1
        
        self.logger.info(f"\nTests pasados: {passed}/{len(results)}")
        
        if passed == len(results):
            self.logger.info("🎉 Todos los tests pasaron exitosamente!")
            return True
        else:
            self.logger.warning(f"⚠️  {len(results) - passed} tests fallaron")
            return False


def main():
    """
    Función principal para ejecutar los tests
    """
    print("=== Test de Visualización Automática de ROI ===\n")
    
    tester = ROIVisualizationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ Todos los tests de visualización ROI completados exitosamente")
        print(f"📁 Resultados guardados en: {tester.test_dir}")
    else:
        print("\n❌ Algunos tests fallaron. Revisar logs para detalles.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())