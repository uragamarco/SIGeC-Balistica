#!/usr/bin/env python3
"""
Test script para validar el sistema con imágenes balísticas reales
Dataset: DeKinder - Sig Sauer P226 9mm Luger
Autor: Equipo de Desarrollo MVP Balístico
Fecha: 2024
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_processing.feature_extractor import extract_orb_features, extract_sift_features
from matching.unified_matcher import UnifiedMatcher
from database.vector_db import VectorDatabase
from config.unified_config import get_unified_config
from utils.logger import get_logger

class RealImageTester:
    """Clase para realizar pruebas con imágenes balísticas reales"""
    
    def __init__(self, dataset_path: str = "uploads/data"):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images" / "dekinder"
        self.metadata_path = self.dataset_path / "processed" / "DeKinder_processed.json"
        
        # Inicializar componentes del sistema
        config = get_unified_config()
        self.db = VectorDatabase(config)
        self.matcher = UnifiedMatcher()
        
        # Configurar logging
        self.logger = get_logger("real_image_test")
        
        # Métricas de prueba
        self.test_results = {
            'total_images': 0,
            'processed_images': 0,
            'failed_extractions': 0,
            'same_weapon_matches': [],
            'different_weapon_matches': [],
            'processing_times': [],
            'accuracy_metrics': {}
        }
        
        # Cargar metadatos
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Cargar metadatos del dataset DeKinder"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.logger.info(f"Metadatos cargados: {len(metadata.get('images', []))} imágenes")
            return metadata
        except Exception as e:
            self.logger.error(f"Error cargando metadatos: {e}")
            return {}
    
    def _get_weapon_id_from_filename(self, filename: str) -> str:
        """Extraer ID del arma del nombre del archivo"""
        # Formato: SS007_CCI BF R.png -> SS007
        return filename.split('_')[0] if '_' in filename else filename.split('.')[0]
    
    def _get_image_type_from_path(self, image_path: Path) -> str:
        """Determinar tipo de imagen basado en la ruta"""
        if 'breech_face' in str(image_path):
            return 'vaina'
        elif 'firing_pin' in str(image_path):
            return 'proyectil'
        else:
            return 'unknown'
    
    def discover_images(self) -> Dict[str, List[Path]]:
        """Descubrir y organizar imágenes por arma"""
        weapon_images = {}
        
        for category in ['breech_face', 'firing_pin']:
            category_path = self.images_path / category
            if not category_path.exists():
                self.logger.warning(f"Directorio no encontrado: {category_path}")
                continue
                
            for image_file in category_path.glob("*.png"):
                weapon_id = self._get_weapon_id_from_filename(image_file.name)
                
                if weapon_id not in weapon_images:
                    weapon_images[weapon_id] = []
                
                weapon_images[weapon_id].append(image_file)
        
        self.logger.info(f"Descubiertas {len(weapon_images)} armas con imágenes")
        for weapon_id, images in weapon_images.items():
            self.logger.info(f"  {weapon_id}: {len(images)} imágenes")
        
        return weapon_images
    
    def test_feature_extraction(self, image_path: Path) -> Optional[Dict]:
        """Probar extracción de características en una imagen real"""
        try:
            start_time = time.time()
            
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image.copy()
            
            # Extraer características usando ORB
            image_type = self._get_image_type_from_path(image_path)
            features = extract_orb_features(gray_img, max_features=500)
            
            processing_time = time.time() - start_time
            
            if features and features.get('num_keypoints', 0) > 0:
                result = {
                    'image_path': str(image_path),
                    'weapon_id': self._get_weapon_id_from_filename(image_path.name),
                    'image_type': image_type,
                    'keypoints_count': features['num_keypoints'],
                    'descriptors_shape': features.get('descriptor_stats', {}),
                    'processing_time': processing_time,
                    'success': True,
                    'features': features
                }
                self.test_results['processed_images'] += 1
            else:
                result = {
                    'image_path': str(image_path),
                    'success': False,
                    'error': 'No se extrajeron características'
                }
                self.test_results['failed_extractions'] += 1
            
            self.test_results['processing_times'].append(processing_time)
            return result
            
        except Exception as e:
            self.logger.error(f"Error procesando {image_path}: {e}")
            self.test_results['failed_extractions'] += 1
            return {
                'image_path': str(image_path),
                'success': False,
                'error': str(e)
            }
    
    def test_same_weapon_matching(self, weapon_images: List[Path], max_comparisons: int = 10) -> List[Dict]:
        """Prueba matching entre imágenes del mismo arma (limitado para optimizar rendimiento)"""
        results = []
        comparisons = 0
        
        if len(weapon_images) < 2:
            self.logger.warning(f"Insuficientes imágenes para prueba de mismo arma: {len(weapon_images)}")
            return results
        
        # Limitar el número de comparaciones para optimizar rendimiento
        for i in range(min(len(weapon_images), 5)):  # Máximo 5 imágenes por arma
            for j in range(i + 1, min(len(weapon_images), i + 3)):  # Máximo 2 comparaciones por imagen
                if comparisons >= max_comparisons:
                    break
                    
                img1_path = weapon_images[i]
                img2_path = weapon_images[j]
                
                try:
                    # Cargar imágenes
                    img1 = cv2.imread(str(img1_path))
                    img2 = cv2.imread(str(img2_path))
                    
                    if img1 is None or img2 is None:
                        continue
                    
                    # Convertir a escala de grises
                    if len(img1.shape) == 3:
                        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    else:
                        gray1 = img1.copy()
                        
                    if len(img2.shape) == 3:
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    else:
                        gray2 = img2.copy()
                    
                    # Extraer características con menos keypoints para optimizar
                    features1 = extract_orb_features(gray1, max_features=200)
                    features2 = extract_orb_features(gray2, max_features=200)
                    
                    if not features1 or not features2:
                        continue
                    
                    if features1.get('num_keypoints', 0) == 0 or features2.get('num_keypoints', 0) == 0:
                        continue
                    
                    # Realizar matching usando el matcher del sistema
                    start_time = time.time()
                    match_result = self.matcher.compare_image_files(img1_path, img2_path)
                    matching_time = time.time() - start_time
                    
                    result = {
                        'image1': img1_path.name,
                        'image2': img2_path.name,
                        'weapon_id': self._get_weapon_id_from_filename(img1_path.name),
                        'similarity': match_result.similarity_score,
                        'good_matches': match_result.good_matches,
                        'total_matches': match_result.total_matches,
                        'matching_time': matching_time,
                        'expected': True  # Mismo arma
                    }
                    
                    results.append(result)
                    comparisons += 1
                    self.logger.info(f"Mismo arma - {img1_path.name} vs {img2_path.name}: {result['similarity']:.2f}%")
                    
                except Exception as e:
                    self.logger.error(f"Error en matching mismo arma {img1_path.name} vs {img2_path.name}: {e}")
            
            if comparisons >= max_comparisons:
                break
        
        return results
    
    def test_matching_different_weapons(self, weapon_images: Dict[str, List[Path]], max_comparisons: int = 15) -> List[Dict]:
        """Prueba matching entre imágenes de armas diferentes (optimizado)"""
        results = []
        comparisons = 0
        
        weapon_ids = list(weapon_images.keys())
        
        for i in range(min(len(weapon_ids), 5)):  # Máximo 5 armas
            for j in range(i + 1, min(len(weapon_ids), i + 4)):  # Máximo 3 comparaciones por arma
                if comparisons >= max_comparisons:
                    break
                    
                weapon1_id = weapon_ids[i]
                weapon2_id = weapon_ids[j]
                
                # Tomar solo 2 imágenes de cada arma para optimizar
                images1 = weapon_images[weapon1_id][:2]
                images2 = weapon_images[weapon2_id][:2]
                
                for img1_path in images1:
                    for img2_path in images2:
                        if comparisons >= max_comparisons:
                            break
                            
                        try:
                            # Cargar imágenes
                            img1 = cv2.imread(str(img1_path))
                            img2 = cv2.imread(str(img2_path))
                            
                            if img1 is None or img2 is None:
                                continue
                            
                            # Convertir a escala de grises
                            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                            
                            # Extraer características con menos keypoints
                            features1 = extract_orb_features(gray1, max_features=200)
                            features2 = extract_orb_features(gray2, max_features=200)
                            
                            if not features1 or not features2:
                                continue
                            
                            if features1.get('num_keypoints', 0) == 0 or features2.get('num_keypoints', 0) == 0:
                                continue
                            
                            # Realizar matching usando el matcher del sistema
                            start_time = time.time()
                            match_result = self.matcher.compare_image_files(img1_path, img2_path)
                            matching_time = time.time() - start_time
                            
                            result = {
                                'image1': img1_path.name,
                                'image2': img2_path.name,
                                'weapon1': weapon1_id,
                                'weapon2': weapon2_id,
                                'similarity': match_result.similarity_score,
                                'good_matches': match_result.good_matches,
                                'total_matches': match_result.total_matches,
                                'matching_time': matching_time,
                                'expected': False  # Diferentes armas
                            }
                            
                            results.append(result)
                            comparisons += 1
                            self.logger.info(f"Diferentes armas - {weapon1_id} vs {weapon2_id}: {result['similarity']:.2f}%")
                            
                        except Exception as e:
                            self.logger.error(f"Error en matching diferentes armas {img1_path.name} vs {img2_path.name}: {e}")
                    
                    if comparisons >= max_comparisons:
                        break
                
                if comparisons >= max_comparisons:
                    break
            
            if comparisons >= max_comparisons:
                break
        
        return results
    
    def calculate_accuracy_metrics(self) -> Dict:
        """Calcular métricas de precisión del sistema"""
        same_weapon_scores = [r['similarity'] for r in self.test_results['same_weapon_matches']]
        different_weapon_scores = [r['similarity'] for r in self.test_results['different_weapon_matches']]
        
        # Definir umbral de decisión (puede ajustarse)
        threshold = 30.0  # 30% de similitud como umbral
        
        # Calcular métricas para mismo arma (verdaderos positivos)
        tp = sum(1 for score in same_weapon_scores if score >= threshold)  # True Positives
        fn = sum(1 for score in same_weapon_scores if score < threshold)   # False Negatives
        
        # Calcular métricas para diferentes armas (verdaderos negativos)
        tn = sum(1 for score in different_weapon_scores if score < threshold)  # True Negatives
        fp = sum(1 for score in different_weapon_scores if score >= threshold) # False Positives
        
        # Métricas de rendimiento
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'threshold': threshold,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'same_weapon_avg_score': np.mean(same_weapon_scores) if same_weapon_scores else 0,
            'different_weapon_avg_score': np.mean(different_weapon_scores) if different_weapon_scores else 0,
            'same_weapon_std': np.std(same_weapon_scores) if same_weapon_scores else 0,
            'different_weapon_std': np.std(different_weapon_scores) if different_weapon_scores else 0
        }
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict:
        """Ejecutar prueba completa con imágenes reales (optimizado)"""
        self.logger.info("=== INICIANDO PRUEBAS CON IMÁGENES REALES (OPTIMIZADO) ===")
        self.logger.info(f"Dataset: {self.dataset_path}")
        
        # Descubrir imágenes
        weapon_images = self.discover_images()
        self.test_results['total_images'] = sum(len(images) for images in weapon_images.values())
        
        if not weapon_images:
            self.logger.error("No se encontraron imágenes para procesar")
            return self.test_results
        
        # Fase 1: Probar extracción de características
        self.logger.info("=== FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS ===")
        extraction_results = []
        
        for weapon_id, images in weapon_images.items():
            self.logger.info(f"Procesando arma {weapon_id}: {len(images)} imágenes")
            for image_path in images:
                result = self.test_feature_extraction(image_path)
                if result:
                    extraction_results.append(result)
        
        # Fase 2: Pruebas de matching mismo arma (optimizado)
        self.logger.info("=== FASE 2: MATCHING MISMO ARMA (OPTIMIZADO) ===")
        for weapon_id, images in weapon_images.items():
            same_weapon_results = self.test_same_weapon_matching(images, max_comparisons=10)
            self.test_results['same_weapon_matches'].extend(same_weapon_results)
        
        # Fase 3: Pruebas de matching diferentes armas (optimizado)
        self.logger.info("=== FASE 3: MATCHING DIFERENTES ARMAS (OPTIMIZADO) ===")
        different_weapon_results = self.test_matching_different_weapons(weapon_images, max_comparisons=15)
        self.test_results['different_weapon_matches'].extend(different_weapon_results)
        
        # Fase 4: Calcular métricas de precisión
        self.logger.info("=== FASE 4: CÁLCULO DE MÉTRICAS ===")
        self.test_results['accuracy_metrics'] = self.calculate_accuracy_metrics()
        
        # Estadísticas finales
        self.test_results['avg_processing_time'] = np.mean(self.test_results['processing_times']) if self.test_results['processing_times'] else 0
        self.test_results['total_comparisons'] = len(self.test_results['same_weapon_matches']) + len(self.test_results['different_weapon_matches'])
        
        self.logger.info("=== PRUEBAS COMPLETADAS ===")
        return self.test_results
    
    def generate_report(self, output_path: str = "reports/real_images_test_report.json"):
        """Generar reporte detallado de las pruebas"""
        report = {
            'test_info': {
                'dataset': 'DeKinder - Sig Sauer P226 9mm Luger',
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_path': str(self.dataset_path),
                'total_images': self.test_results['total_images'],
                'processed_images': self.test_results['processed_images'],
                'failed_extractions': self.test_results['failed_extractions']
            },
            'performance_metrics': {
                'avg_processing_time': self.test_results['avg_processing_time'],
                'total_comparisons': self.test_results['total_comparisons'],
                'same_weapon_comparisons': len(self.test_results['same_weapon_matches']),
                'different_weapon_comparisons': len(self.test_results['different_weapon_matches'])
            },
            'accuracy_metrics': self.test_results['accuracy_metrics'],
            'detailed_results': {
                'same_weapon_matches': self.test_results['same_weapon_matches'],
                'different_weapon_matches': self.test_results['different_weapon_matches']
            }
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Reporte guardado en: {output_path}")
        return report

def main():
    """Función principal para ejecutar las pruebas"""
    print("=== SISTEMA DE PRUEBAS CON IMÁGENES BALÍSTICAS REALES ===")
    print("Dataset: DeKinder - Sig Sauer P226 9mm Luger")
    print()
    
    # Crear instancia del tester
    tester = RealImageTester()
    
    # Ejecutar pruebas
    results = tester.run_comprehensive_test()
    
    # Generar reporte
    report = tester.generate_report()
    
    # Mostrar resumen
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"Imágenes totales: {results['total_images']}")
    print(f"Imágenes procesadas: {results['processed_images']}")
    print(f"Extracciones fallidas: {results['failed_extractions']}")
    print(f"Tiempo promedio de procesamiento: {results['avg_processing_time']:.3f}s")
    print(f"Total de comparaciones: {results['total_comparisons']}")
    
    if results['accuracy_metrics']:
        metrics = results['accuracy_metrics']
        print(f"\n=== MÉTRICAS DE PRECISIÓN ===")
        print(f"Umbral de decisión: {metrics['threshold']}%")
        print(f"Precisión: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"Exactitud: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1_score']:.3f}")
        print(f"Similitud promedio mismo arma: {metrics['same_weapon_avg_score']:.2f}%")
        print(f"Similitud promedio diferentes armas: {metrics['different_weapon_avg_score']:.2f}%")
    
    print(f"\nReporte detallado guardado en: reports/real_images_test_report.json")
    
    return results

if __name__ == "__main__":
    main()