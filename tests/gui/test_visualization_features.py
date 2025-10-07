#!/usr/bin/env python3
"""
Script de Prueba Integral para Funciones de Visualización de Características
Sistema Balístico Forense MVP v0.1.3

Este script prueba todas las nuevas funciones de visualización implementadas en:
- feature_extractor.py
- unified_preprocessor.py  
- unified_roi_detector.py

Autor: Sistema de Pruebas Automatizadas
Fecha: 2025-09-30
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar módulos del sistema
try:
    from image_processing.feature_extractor import (
        visualize_keypoints,
        visualize_sift_features,
        visualize_orb_features,
        visualize_lbp_texture,
        visualize_gabor_filters,
        visualize_ballistic_features,
        create_feature_comparison_visualization,
        extract_sift_features,
        extract_orb_features,
        extract_advanced_lbp,
        extract_gabor_features,
        calculate_ballistic_features
    )
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importando feature_extractor: {e}")
    FEATURE_EXTRACTOR_AVAILABLE = False

try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importando unified_preprocessor: {e}")
    PREPROCESSOR_AVAILABLE = False

try:
    from image_processing.unified_roi_detector import visualize_roi
    # Intentar importar detect_roi, pero puede no estar disponible
    try:
        from image_processing.unified_roi_detector import detect_roi
        ROI_DETECT_AVAILABLE = True
    except ImportError:
        ROI_DETECT_AVAILABLE = False
    ROI_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importando unified_roi_detector: {e}")
    ROI_DETECTOR_AVAILABLE = False
    ROI_DETECT_AVAILABLE = False

class VisualizationTester:
    """Clase para probar todas las funciones de visualización"""
    
    def __init__(self):
        self.test_results = {}
        self.output_dir = Path("temp/test_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear imagen de prueba sintética
        self.test_image = self._create_test_image()
        
    def _create_test_image(self):
        """Crear una imagen de prueba sintética con características detectables"""
        # Crear imagen base
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Agregar formas geométricas para generar características
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
        cv2.rectangle(img, (200, 50), (350, 150), (128, 128, 128), -1)
        cv2.ellipse(img, (100, 300), (80, 40), 45, 0, 360, (200, 200, 200), -1)
        
        # Agregar líneas para crear bordes
        cv2.line(img, (0, 200), (400, 200), (255, 255, 255), 2)
        cv2.line(img, (200, 0), (200, 400), (255, 255, 255), 2)
        
        # Agregar ruido para simular textura
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def test_feature_extractor_visualizations(self):
        """Probar todas las funciones de visualización del feature_extractor"""
        logger.info("=== Probando Funciones de Visualización de Características ===")
        
        if not FEATURE_EXTRACTOR_AVAILABLE:
            logger.error("Feature extractor no disponible, saltando pruebas")
            return
        
        try:
            # 1. Probar visualización de keypoints genérica con keypoints simulados
            logger.info("1. Probando visualize_keypoints...")
            # Crear keypoints simulados
            keypoints = [cv2.KeyPoint(x=50, y=50, size=10), cv2.KeyPoint(x=100, y=100, size=15)]
            keypoints_img = visualize_keypoints(self.test_image, keypoints, algorithm='TEST')
            if keypoints_img is not None:
                cv2.imwrite(str(self.output_dir / "test_keypoints.jpg"), keypoints_img)
                self.test_results['visualize_keypoints'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_keypoints: EXITOSO")
            else:
                self.test_results['visualize_keypoints'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_keypoints: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_keypoints'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_keypoints: ERROR - {str(e)}")
        
        try:
            # 2. Probar visualización SIFT
            logger.info("2. Probando visualize_sift_features...")
            # Convertir a escala de grises si es necesario
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            sift_img, features = visualize_sift_features(gray_img)
            if sift_img is not None:
                cv2.imwrite(str(self.output_dir / "test_sift.jpg"), sift_img)
                self.test_results['visualize_sift_features'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_sift_features: EXITOSO")
            else:
                self.test_results['visualize_sift_features'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_sift_features: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_sift_features'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_sift_features: ERROR - {str(e)}")
        
        try:
            # 3. Probar visualización ORB
            logger.info("3. Probando visualize_orb_features...")
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            orb_img, features = visualize_orb_features(gray_img)
            if orb_img is not None:
                cv2.imwrite(str(self.output_dir / "test_orb.jpg"), orb_img)
                self.test_results['visualize_orb_features'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_orb_features: EXITOSO")
            else:
                self.test_results['visualize_orb_features'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_orb_features: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_orb_features'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_orb_features: ERROR - {str(e)}")
        
        try:
            # 4. Probar visualización LBP
            logger.info("4. Probando visualize_lbp_texture...")
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            lbp_img, features = visualize_lbp_texture(gray_img)
            if lbp_img is not None:
                cv2.imwrite(str(self.output_dir / "test_lbp.jpg"), lbp_img)
                self.test_results['visualize_lbp_texture'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_lbp_texture: EXITOSO")
            else:
                self.test_results['visualize_lbp_texture'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_lbp_texture: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_lbp_texture'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_lbp_texture: ERROR - {str(e)}")
        
        try:
            # 5. Probar visualización Gabor
            logger.info("5. Probando visualize_gabor_filters...")
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            gabor_img, features = visualize_gabor_filters(gray_img)
            if gabor_img is not None:
                cv2.imwrite(str(self.output_dir / "test_gabor.jpg"), gabor_img)
                self.test_results['visualize_gabor_filters'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_gabor_filters: EXITOSO")
            else:
                self.test_results['visualize_gabor_filters'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_gabor_filters: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_gabor_filters'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_gabor_filters: ERROR - {str(e)}")
        
        try:
            # 6. Probar visualización de características balísticas
            logger.info("6. Probando visualize_ballistic_features...")
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            ballistic_img, features = visualize_ballistic_features(gray_img)
            if ballistic_img is not None:
                cv2.imwrite(str(self.output_dir / "test_ballistic.jpg"), ballistic_img)
                self.test_results['visualize_ballistic_features'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_ballistic_features: EXITOSO")
            else:
                self.test_results['visualize_ballistic_features'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_ballistic_features: FALLÓ")
                
        except Exception as e:
            self.test_results['visualize_ballistic_features'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_ballistic_features: ERROR - {str(e)}")
        
        try:
            # 7. Probar visualización comparativa
            logger.info("7. Probando create_feature_comparison_visualization...")
            gray_img = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY) if len(self.test_image.shape) == 3 else self.test_image
            comparison_img = create_feature_comparison_visualization(gray_img, algorithms=['SIFT', 'ORB'])
            if comparison_img is not None:
                cv2.imwrite(str(self.output_dir / "test_comparison.jpg"), comparison_img)
                self.test_results['create_feature_comparison_visualization'] = "✓ EXITOSO"
                logger.info("   ✓ create_feature_comparison_visualization: EXITOSO")
            else:
                self.test_results['create_feature_comparison_visualization'] = "✗ FALLÓ"
                logger.error("   ✗ create_feature_comparison_visualization: FALLÓ")
                
        except Exception as e:
            self.test_results['create_feature_comparison_visualization'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ create_feature_comparison_visualization: ERROR - {str(e)}")
    
    def test_preprocessor_visualizations(self):
        """Probar visualizaciones del preprocesador"""
        logger.info("=== Probando Funciones de Visualización del Preprocesador ===")
        
        if not PREPROCESSOR_AVAILABLE:
            logger.error("Preprocessor no disponible, saltando pruebas")
            return
        
        try:
            logger.info("Probando preprocess_with_visualization...")
            
            # Crear instancia del preprocesador con visualización habilitada
            preprocessor = UnifiedPreprocessor()
            preprocessor.config.enable_visualization = True
            preprocessor.config.save_intermediate_steps = True
            preprocessor.config.visualization_output_dir = str(self.output_dir)
            
            # Guardar imagen de prueba temporalmente
            temp_image_path = str(self.output_dir / "temp_test_image.jpg")
            cv2.imwrite(temp_image_path, self.test_image)
            
            # Procesar con visualización
            result = preprocessor.preprocess_with_visualization(temp_image_path)
            
            if result.success and result.processed_image is not None:
                # Guardar imagen procesada
                cv2.imwrite(str(self.output_dir / "test_preprocessed.jpg"), result.processed_image)
                
                # Verificar imágenes intermedias
                if result.intermediate_images:
                    for step_name, step_img in result.intermediate_images.items():
                        if step_img is not None:
                            cv2.imwrite(str(self.output_dir / f"test_preprocess_{step_name}.jpg"), step_img)
                
                self.test_results['preprocess_with_visualization'] = "✓ EXITOSO"
                logger.info("   ✓ preprocess_with_visualization: EXITOSO")
            else:
                self.test_results['preprocess_with_visualization'] = f"✗ FALLÓ - {result.error_message}"
                logger.error(f"   ✗ preprocess_with_visualization: FALLÓ - {result.error_message}")
                
            # Limpiar archivo temporal
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
        except Exception as e:
            self.test_results['preprocess_with_visualization'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ preprocess_with_visualization: ERROR - {str(e)}")
    
    def test_roi_visualizations(self):
        """Probar visualizaciones del detector ROI"""
        logger.info("=== Probando Funciones de Visualización del Detector ROI ===")
        
        if not ROI_DETECTOR_AVAILABLE:
            logger.error("ROI detector no disponible, saltando pruebas")
            return
        
        try:
            logger.info("Probando visualize_roi...")
            
            # Crear datos ROI simulados para la prueba
            roi_results = {
                'roi_coordinates': [(50, 50, 200, 200)],  # x, y, width, height
                'confidence': [0.95]
            }
            
            # Visualizar ROI
            roi_img = visualize_roi(self.test_image, roi_results)
            
            if roi_img is not None:
                cv2.imwrite(str(self.output_dir / "test_roi.jpg"), roi_img)
                self.test_results['visualize_roi'] = "✓ EXITOSO"
                logger.info("   ✓ visualize_roi: EXITOSO")
            else:
                self.test_results['visualize_roi'] = "✗ FALLÓ"
                logger.error("   ✗ visualize_roi: FALLÓ")
                
            # Si detect_roi está disponible, probarlo también
            if ROI_DETECT_AVAILABLE:
                logger.info("Probando detect_roi...")
                roi_results_detected = detect_roi(self.test_image)
                if roi_results_detected:
                    self.test_results['detect_roi'] = "✓ EXITOSO"
                    logger.info("   ✓ detect_roi: EXITOSO")
                else:
                    self.test_results['detect_roi'] = "✗ FALLÓ"
                    logger.error("   ✗ detect_roi: FALLÓ")
            else:
                logger.info("   detect_roi no disponible, usando datos simulados")
                
        except Exception as e:
            self.test_results['visualize_roi'] = f"✗ ERROR: {str(e)}"
            logger.error(f"   ✗ visualize_roi: ERROR - {str(e)}")
    
    def generate_test_report(self):
        """Generar reporte de resultados de las pruebas"""
        logger.info("=== Generando Reporte de Pruebas ===")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.startswith("✓"))
        failed_tests = total_tests - successful_tests
        
        report = f"""
REPORTE DE PRUEBAS INTEGRALES DE VISUALIZACIÓN
==============================================
Fecha: 2025-09-30
Sistema: Balístico Forense MVP v0.1.3

RESUMEN:
--------
Total de pruebas: {total_tests}
Exitosas: {successful_tests}
Fallidas: {failed_tests}
Tasa de éxito: {(successful_tests/total_tests)*100:.1f}%

RESULTADOS DETALLADOS:
---------------------
"""
        
        for function_name, result in self.test_results.items():
            report += f"{function_name}: {result}\n"
        
        report += f"""
ARCHIVOS GENERADOS:
------------------
Directorio de salida: {self.output_dir}
"""
        
        # Listar archivos generados
        for file_path in self.output_dir.glob("*.jpg"):
            report += f"- {file_path.name}\n"
        
        # Guardar reporte
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Reporte guardado en: {report_path}")
        print(report)
        
        return successful_tests, failed_tests

def main():
    """Función principal para ejecutar todas las pruebas"""
    logger.info("Iniciando Pruebas Integrales de Visualización")
    
    tester = VisualizationTester()
    
    # Ejecutar todas las pruebas
    tester.test_feature_extractor_visualizations()
    tester.test_preprocessor_visualizations()
    tester.test_roi_visualizations()
    
    # Generar reporte
    successful, failed = tester.generate_test_report()
    
    if failed == 0:
        logger.info("🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        return 0
    else:
        logger.warning(f"⚠️  {failed} pruebas fallaron. Revisar el reporte para detalles.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)