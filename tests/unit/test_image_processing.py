#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Pruebas Integral para Módulos de Procesamiento de Imágenes
Sistema Balístico Forense SIGeC-Balistica

Este script realiza pruebas exhaustivas de todos los módulos de procesamiento
de imágenes utilizando las muestras NIST FADB disponibles.

Módulos a probar:
- feature_extractor.py
- ballistic_features.py
- feature_visualizer.py
- preprocessing_visualizer.py
- statistical_analyzer.py
- statistical_visualizer.py
- unified_preprocessor.py
- unified_roi_detector.py
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Agregar el directorio raíz al path
sys.path.append('/home/marco/SIGeC-Balistica')

# Importar módulos del sistema
try:
    from image_processing.feature_extractor import FeatureExtractor, extract_features
    from image_processing.ballistic_features import BallisticFeatureExtractor, extract_ballistic_features_from_path
    from image_processing.feature_visualizer import FeatureVisualizer
    from image_processing.preprocessing_visualizer import PreprocessingVisualizer
    from image_processing.statistical_analyzer import StatisticalAnalyzer
    from image_processing.statistical_visualizer import StatisticalVisualizer
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingLevel
    from image_processing.unified_roi_detector import UnifiedROIDetector
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importando módulos: {e}")
    MODULES_AVAILABLE = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ImageProcessingTest')

class ImageProcessingTester:
    """Clase principal para realizar pruebas de los módulos de procesamiento"""
    
    def __init__(self, samples_dir: str = "uploads/Muestras NIST FADB"):
        self.samples_dir = Path(samples_dir)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Crear subdirectorios para resultados
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "features").mkdir(exist_ok=True)
        (self.results_dir / "preprocessing").mkdir(exist_ok=True)
        (self.results_dir / "roi_detection").mkdir(exist_ok=True)
        (self.results_dir / "statistical").mkdir(exist_ok=True)
        
        # Inicializar módulos
        self.feature_extractor = FeatureExtractor()
        self.ballistic_extractor = BallisticFeatureExtractor()
        self.feature_visualizer = FeatureVisualizer(str(self.results_dir / "visualizations"))
        self.preprocessing_visualizer = PreprocessingVisualizer()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.statistical_visualizer = StatisticalVisualizer(str(self.results_dir / "statistical"))
        self.preprocessor = UnifiedPreprocessor()
        self.roi_detector = UnifiedROIDetector()
        
        # Resultados de pruebas
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'modules_tested': [],
            'images_processed': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'performance_metrics': {},
            'feature_statistics': {},
            'module_status': {}
        }
        
        logger.info(f"Tester inicializado. Directorio de muestras: {self.samples_dir}")
        logger.info(f"Directorio de resultados: {self.results_dir}")
    
    def find_sample_images(self, max_images: int = 20) -> List[Path]:
        """Encuentra imágenes de muestra para las pruebas"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        sample_images = []
        
        try:
            for root, dirs, files in os.walk(self.samples_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_path = Path(root) / file
                        sample_images.append(image_path)
                        
                        if len(sample_images) >= max_images:
                            break
                
                if len(sample_images) >= max_images:
                    break
            
            logger.info(f"Encontradas {len(sample_images)} imágenes de muestra")
            return sample_images[:max_images]
            
        except Exception as e:
            logger.error(f"Error buscando imágenes de muestra: {e}")
            return []
    
    def test_feature_extraction(self, image_path: Path) -> Dict[str, Any]:
        """Prueba la extracción de características"""
        test_result = {
            'module': 'feature_extractor',
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'features_extracted': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Extraer características usando el módulo principal
            features = extract_features(str(image_path))
            
            # Extraer características usando la clase FeatureExtractor
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sift_features = self.feature_extractor.extract_sift_features(gray_image)
            orb_features = self.feature_extractor.extract_orb_features(gray_image)
            lbp_features = self.feature_extractor.extract_lbp_features(gray_image)
            gabor_features = self.feature_extractor.extract_gabor_features(gray_image)
            
            processing_time = time.time() - start_time
            
            # Contar características extraídas
            features_count = 0
            if 'sift_keypoints' in sift_features:
                features_count += len(sift_features['sift_keypoints'])
            if 'orb_keypoints' in orb_features:
                features_count += len(orb_features['orb_keypoints'])
            
            test_result.update({
                'success': True,
                'processing_time': processing_time,
                'features_extracted': features_count,
                'sift_keypoints': len(sift_features.get('sift_keypoints', [])),
                'orb_keypoints': len(orb_features.get('orb_keypoints', [])),
                'lbp_histogram_size': len(lbp_features.get('lbp_histogram', [])),
                'gabor_responses': len(gabor_features.get('gabor_responses', []))
            })
            
            # Guardar características
            features_file = self.results_dir / "features" / f"{image_path.stem}_features.json"
            with open(features_file, 'w') as f:
                json.dump({
                    'main_features': features,
                    'sift_features': sift_features,
                    'orb_features': orb_features,
                    'lbp_features': lbp_features,
                    'gabor_features': gabor_features
                }, f, indent=2, default=str)
            
            logger.info(f"Extracción de características exitosa para {image_path.name}")
            
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"Error en extracción de características para {image_path.name}: {e}")
        
        return test_result
    
    def test_ballistic_features(self, image_path: Path) -> Dict[str, Any]:
        """Prueba la extracción de características balísticas"""
        test_result = {
            'module': 'ballistic_features',
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Extraer características balísticas
            ballistic_features = extract_ballistic_features_from_path(str(image_path))
            
            processing_time = time.time() - start_time
            
            if ballistic_features and ballistic_features.get('success', False):
                test_result.update({
                    'success': True,
                    'processing_time': processing_time,
                    'quality_score': ballistic_features.get('quality_metrics', {}).get('quality_score', 0),
                    'confidence': ballistic_features.get('quality_metrics', {}).get('confidence', 0),
                    'roi_regions_detected': len(ballistic_features.get('roi_regions', []))
                })
                
                # Guardar características balísticas
                ballistic_file = self.results_dir / "features" / f"{image_path.stem}_ballistic.json"
                with open(ballistic_file, 'w') as f:
                    json.dump(ballistic_features, f, indent=2, default=str)
                
                logger.info(f"Extracción de características balísticas exitosa para {image_path.name}")
            else:
                test_result['error'] = "No se pudieron extraer características balísticas"
                
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"Error en características balísticas para {image_path.name}: {e}")
        
        return test_result
    
    def test_preprocessing(self, image_path: Path) -> Dict[str, Any]:
        """Prueba el preprocesamiento de imágenes"""
        test_result = {
            'module': 'preprocessing',
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Preprocesar imagen con visualización
            preprocessing_result = self.preprocessing_visualizer.preprocess_with_visualization(
                str(image_path),
                evidence_type="cartridge_case",
                level="standard",
                output_dir=str(self.results_dir / "preprocessing"),
                save_steps=True
            )
            
            processing_time = time.time() - start_time
            
            if preprocessing_result.success:
                test_result.update({
                    'success': True,
                    'processing_time': processing_time,
                    'steps_applied': len(preprocessing_result.steps),
                    'visualization_saved': preprocessing_result.visualization_path is not None
                })
                
                logger.info(f"Preprocesamiento exitoso para {image_path.name}")
            else:
                test_result['error'] = preprocessing_result.error_message
                
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"Error en preprocesamiento para {image_path.name}: {e}")
        
        return test_result
    
    def test_roi_detection(self, image_path: Path) -> Dict[str, Any]:
        """Prueba la detección de ROI"""
        test_result = {
            'module': 'roi_detection',
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Detectar ROI
            roi_regions = self.roi_detector.detect_roi(
                str(image_path),
                specimen_type='cartridge_case',
                level='standard'
            )
            
            processing_time = time.time() - start_time
            
            test_result.update({
                'success': True,
                'processing_time': processing_time,
                'roi_regions_detected': len(roi_regions),
                'avg_confidence': np.mean([r.confidence for r in roi_regions]) if roi_regions else 0
            })
            
            # Guardar resultados de ROI
            roi_file = self.results_dir / "roi_detection" / f"{image_path.stem}_roi.json"
            roi_data = [region.to_dict() for region in roi_regions]
            with open(roi_file, 'w') as f:
                json.dump(roi_data, f, indent=2, default=str)
            
            # Crear visualización de ROI
            if roi_regions:
                image = self.roi_detector.load_image(str(image_path))
                if image is not None:
                    roi_vis = self.roi_detector.visualize_roi(image, roi_regions)
                    roi_vis_path = self.results_dir / "roi_detection" / f"{image_path.stem}_roi_vis.png"
                    cv2.imwrite(str(roi_vis_path), cv2.cvtColor(roi_vis, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Detección de ROI exitosa para {image_path.name}: {len(roi_regions)} regiones")
            
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"Error en detección de ROI para {image_path.name}: {e}")
        
        return test_result
    
    def test_feature_visualization(self, image_path: Path) -> Dict[str, Any]:
        """Prueba la visualización de características"""
        test_result = {
            'module': 'feature_visualization',
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraer keypoints para visualización
            sift = cv2.SIFT_create(nfeatures=100)
            orb = cv2.ORB_create(nfeatures=100)
            
            sift_kp, _ = sift.detectAndCompute(gray_image, None)
            orb_kp, _ = orb.detectAndCompute(gray_image, None)
            
            # Crear visualizaciones
            vis_sift = self.feature_visualizer.visualize_keypoints(
                gray_image, sift_kp, "SIFT",
                str(self.results_dir / "visualizations" / f"{image_path.stem}_sift.png")
            )
            
            vis_orb = self.feature_visualizer.visualize_keypoints(
                gray_image, orb_kp, "ORB",
                str(self.results_dir / "visualizations" / f"{image_path.stem}_orb.png")
            )
            
            # Visualizar texturas LBP
            lbp_vis = self.feature_visualizer.visualize_lbp_texture(
                gray_image,
                save_path=str(self.results_dir / "visualizations" / f"{image_path.stem}_lbp.png")
            )
            
            # Crear comparación de características
            self.feature_visualizer.create_feature_comparison(
                gray_image, sift_kp, orb_kp,
                str(self.results_dir / "visualizations" / f"{image_path.stem}_comparison.png")
            )
            
            processing_time = time.time() - start_time
            
            test_result.update({
                'success': True,
                'processing_time': processing_time,
                'sift_keypoints': len(sift_kp),
                'orb_keypoints': len(orb_kp),
                'visualizations_created': 4
            })
            
            logger.info(f"Visualización de características exitosa para {image_path.name}")
            
        except Exception as e:
            test_result['error'] = str(e)
            logger.error(f"Error en visualización de características para {image_path.name}: {e}")
        
        return test_result
    
    def run_comprehensive_test(self, max_images: int = 10) -> Dict[str, Any]:
        """Ejecuta pruebas comprehensivas en todas las imágenes de muestra"""
        logger.info("Iniciando pruebas comprehensivas del sistema de procesamiento de imágenes")
        
        # Encontrar imágenes de muestra
        sample_images = self.find_sample_images(max_images)
        
        if not sample_images:
            logger.error("No se encontraron imágenes de muestra")
            return self.test_results
        
        logger.info(f"Procesando {len(sample_images)} imágenes de muestra")
        
        # Ejecutar pruebas para cada imagen
        for i, image_path in enumerate(sample_images, 1):
            logger.info(f"Procesando imagen {i}/{len(sample_images)}: {image_path.name}")
            
            # Prueba de extracción de características
            feature_result = self.test_feature_extraction(image_path)
            self.test_results['modules_tested'].append(feature_result)
            
            # Prueba de características balísticas
            ballistic_result = self.test_ballistic_features(image_path)
            self.test_results['modules_tested'].append(ballistic_result)
            
            # Prueba de preprocesamiento
            preprocessing_result = self.test_preprocessing(image_path)
            self.test_results['modules_tested'].append(preprocessing_result)
            
            # Prueba de detección de ROI
            roi_result = self.test_roi_detection(image_path)
            self.test_results['modules_tested'].append(roi_result)
            
            # Prueba de visualización
            visualization_result = self.test_feature_visualization(image_path)
            self.test_results['modules_tested'].append(visualization_result)
            
            # Actualizar contadores
            self.test_results['images_processed'] += 1
            
            # Contar éxitos y fallos
            for result in [feature_result, ballistic_result, preprocessing_result, roi_result, visualization_result]:
                if result['success']:
                    self.test_results['successful_tests'] += 1
                else:
                    self.test_results['failed_tests'] += 1
                    if result['error']:
                        self.test_results['errors'].append({
                            'module': result['module'],
                            'image': result['image'],
                            'error': result['error']
                        })
        
        # Calcular estadísticas finales
        self.calculate_final_statistics()
        
        # Guardar resultados
        self.save_test_results()
        
        logger.info("Pruebas comprehensivas completadas")
        return self.test_results
    
    def calculate_final_statistics(self):
        """Calcula estadísticas finales de las pruebas"""
        modules = {}
        
        for test in self.test_results['modules_tested']:
            module = test['module']
            if module not in modules:
                modules[module] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'failed_tests': 0,
                    'avg_processing_time': 0,
                    'processing_times': []
                }
            
            modules[module]['total_tests'] += 1
            if test['success']:
                modules[module]['successful_tests'] += 1
            else:
                modules[module]['failed_tests'] += 1
            
            if 'processing_time' in test:
                modules[module]['processing_times'].append(test['processing_time'])
        
        # Calcular promedios
        for module, stats in modules.items():
            if stats['processing_times']:
                stats['avg_processing_time'] = np.mean(stats['processing_times'])
                stats['success_rate'] = stats['successful_tests'] / stats['total_tests']
        
        self.test_results['module_status'] = modules
    
    def save_test_results(self):
        """Guarda los resultados de las pruebas"""
        results_file = self.results_dir / "comprehensive_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Crear reporte en texto
        report_file = self.results_dir / "test_report.txt"
        with open(report_file, 'w') as f:
            f.write("REPORTE DE PRUEBAS DEL SISTEMA DE PROCESAMIENTO DE IMÁGENES\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha: {self.test_results['timestamp']}\n")
            f.write(f"Imágenes procesadas: {self.test_results['images_processed']}\n")
            f.write(f"Pruebas exitosas: {self.test_results['successful_tests']}\n")
            f.write(f"Pruebas fallidas: {self.test_results['failed_tests']}\n")
            f.write(f"Tasa de éxito general: {self.test_results['successful_tests']/(self.test_results['successful_tests']+self.test_results['failed_tests'])*100:.1f}%\n\n")
            
            f.write("ESTADO POR MÓDULO:\n")
            f.write("-" * 30 + "\n")
            for module, stats in self.test_results['module_status'].items():
                f.write(f"{module}:\n")
                f.write(f"  Pruebas totales: {stats['total_tests']}\n")
                f.write(f"  Éxitos: {stats['successful_tests']}\n")
                f.write(f"  Fallos: {stats['failed_tests']}\n")
                f.write(f"  Tasa de éxito: {stats.get('success_rate', 0)*100:.1f}%\n")
                f.write(f"  Tiempo promedio: {stats.get('avg_processing_time', 0):.3f}s\n\n")
            
            if self.test_results['errors']:
                f.write("ERRORES ENCONTRADOS:\n")
                f.write("-" * 30 + "\n")
                for error in self.test_results['errors']:
                    f.write(f"Módulo: {error['module']}\n")
                    f.write(f"Imagen: {error['image']}\n")
                    f.write(f"Error: {error['error']}\n\n")
        
        logger.info(f"Resultados guardados en: {results_file}")
        logger.info(f"Reporte guardado en: {report_file}")

def main():
    """Función principal"""
    if not MODULES_AVAILABLE:
        print("Error: No se pudieron importar los módulos necesarios")
        return
    
    print("Iniciando pruebas del sistema de procesamiento de imágenes...")
    print("=" * 60)
    
    # Crear tester
    tester = ImageProcessingTester()
    
    # Ejecutar pruebas
    results = tester.run_comprehensive_test(max_images=15)
    
    # Mostrar resumen
    print("\nRESUMEN DE PRUEBAS:")
    print("-" * 30)
    print(f"Imágenes procesadas: {results['images_processed']}")
    print(f"Pruebas exitosas: {results['successful_tests']}")
    print(f"Pruebas fallidas: {results['failed_tests']}")
    
    if results['successful_tests'] + results['failed_tests'] > 0:
        success_rate = results['successful_tests'] / (results['successful_tests'] + results['failed_tests']) * 100
        print(f"Tasa de éxito: {success_rate:.1f}%")
    
    print(f"\nResultados detallados en: {tester.results_dir}")
    
    # Mostrar estado por módulo
    print("\nESTADO POR MÓDULO:")
    print("-" * 30)
    for module, stats in results['module_status'].items():
        success_rate = stats.get('success_rate', 0) * 100
        avg_time = stats.get('avg_processing_time', 0)
        print(f"{module}: {success_rate:.1f}% éxito, {avg_time:.3f}s promedio")

if __name__ == "__main__":
    main()