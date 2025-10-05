#!/usr/bin/env python3
"""
Script de prueba para el matcher mejorado
"""

import cv2
import numpy as np
import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from matching.unified_matcher import UnifiedMatcher
from utils.logger import get_logger

class ImprovedMatchingTester:
    def __init__(self, dataset_path: str = "uploads/data"):
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__)
        self.matcher = UnifiedMatcher()
        
    def discover_images(self):
        """Descubrir imágenes agrupadas por arma"""
        weapon_images = defaultdict(list)
        
        # Buscar en el dataset final
        dataset_dir = self.dataset_path / "final_dataset"
        if not dataset_dir.exists():
            self.logger.error(f"Dataset no encontrado: {dataset_dir}")
            return weapon_images
        
        # Buscar imágenes en todas las subcarpetas
        for image_path in dataset_dir.rglob("*.png"):
            weapon_id = self._get_weapon_id_from_filename(image_path.name)
            if weapon_id and weapon_id != "unknown":
                weapon_images[weapon_id].append(image_path)
        
        return weapon_images
    
    def _get_weapon_id_from_filename(self, filename: str) -> str:
        """Extraer ID del arma del nombre del archivo"""
        parts = filename.split('_')
        
        # Para archivos DeKinder: DeKinder_SS007_...
        if parts[0] == "DeKinder" and len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # DeKinder_SS007
        
        # Para archivos PopStat: PopStat_Colt_... o PopStat_Ruger_...
        elif parts[0] == "PopStat" and len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # PopStat_Colt, PopStat_Ruger
        
        # Para otros patrones (Cary_Persistence_...)
        elif len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        
        # Fallback
        elif len(parts) >= 1:
            return parts[0]
        
        return "unknown"
    
    def test_same_weapon_matching(self, weapon_images: dict, max_comparisons: int = 50):
        """Probar matching entre imágenes del mismo arma"""
        results = []
        comparisons = 0
        
        self.logger.info("=== PRUEBAS MISMO ARMA ===")
        
        for weapon_id, images in weapon_images.items():
            if len(images) < 2:
                continue
                
            self.logger.info(f"Probando arma: {weapon_id} ({len(images)} imágenes)")
            
            # Comparar pares de imágenes del mismo arma
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    if comparisons >= max_comparisons:
                        break
                    
                    img1_path = images[i]
                    img2_path = images[j]
                    
                    result = self.matcher.compare_image_files(str(img1_path), str(img2_path))
                    
                    # Convertir MatchResult a dict para compatibilidad
                    result_dict = {
                        'similarity': result.similarity_score,
                        'confidence': result.confidence,
                        'total_matches': result.total_matches,
                        'good_matches': result.good_matches,
                        'processing_time': result.processing_time,
                        'weapon_id': weapon_id,
                        'comparison_type': 'same_weapon',
                        'image1_name': img1_path.name,
                        'image2_name': img2_path.name
                    }
                    
                    results.append(result_dict)
                    comparisons += 1
                    
                    self.logger.info(f"  {img1_path.name} vs {img2_path.name}: {result.similarity_score:.2f}%")
                
                if comparisons >= max_comparisons:
                    break
            
            if comparisons >= max_comparisons:
                break
        
        return results
    
    def test_different_weapon_matching(self, weapon_images: dict, max_comparisons: int = 50):
        """Probar matching entre imágenes de diferentes armas"""
        results = []
        comparisons = 0
        
        self.logger.info("=== PRUEBAS DIFERENTES ARMAS ===")
        
        weapon_list = list(weapon_images.keys())
        
        for i in range(len(weapon_list)):
            for j in range(i + 1, len(weapon_list)):
                if comparisons >= max_comparisons:
                    break
                
                weapon1 = weapon_list[i]
                weapon2 = weapon_list[j]
                
                images1 = weapon_images[weapon1]
                images2 = weapon_images[weapon2]
                
                if len(images1) == 0 or len(images2) == 0:
                    continue
                
                # Tomar una imagen de cada arma
                img1_path = images1[0]
                img2_path = images2[0]
                
                result = self.matcher.compare_image_files(str(img1_path), str(img2_path))
                
                # Convertir MatchResult a dict para compatibilidad
                result_dict = {
                    'similarity': result.similarity_score,
                    'confidence': result.confidence,
                    'total_matches': result.total_matches,
                    'good_matches': result.good_matches,
                    'processing_time': result.processing_time,
                    'weapon1_id': weapon1,
                    'weapon2_id': weapon2,
                    'comparison_type': 'different_weapon',
                    'image1_name': img1_path.name,
                    'image2_name': img2_path.name
                }
                
                results.append(result_dict)
                comparisons += 1
                
                self.logger.info(f"  {weapon1} vs {weapon2}: {result.similarity_score:.2f}%")
            
            if comparisons >= max_comparisons:
                break
        
        return results
    
    def calculate_metrics(self, same_weapon_results: list, different_weapon_results: list, threshold: float = 20.0):
        """Calcular métricas de rendimiento"""
        # Extraer similitudes
        same_similarities = [r['similarity'] for r in same_weapon_results]
        diff_similarities = [r['similarity'] for r in different_weapon_results]
        
        # Clasificación binaria
        same_predictions = [1 if sim >= threshold else 0 for sim in same_similarities]
        diff_predictions = [1 if sim >= threshold else 0 for sim in diff_similarities]
        
        # Ground truth
        same_labels = [1] * len(same_similarities)  # Mismo arma = positivo
        diff_labels = [0] * len(diff_similarities)  # Diferente arma = negativo
        
        # Calcular métricas
        tp = sum(same_predictions)  # Mismo arma correctamente identificado
        fn = len(same_predictions) - tp  # Mismo arma incorrectamente rechazado
        fp = sum(diff_predictions)  # Diferente arma incorrectamente aceptado
        tn = len(diff_predictions) - fp  # Diferente arma correctamente rechazado
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'threshold': threshold,
            'same_weapon_stats': {
                'mean': np.mean(same_similarities),
                'std': np.std(same_similarities),
                'min': np.min(same_similarities),
                'max': np.max(same_similarities),
                'count': len(same_similarities)
            },
            'different_weapon_stats': {
                'mean': np.mean(diff_similarities),
                'std': np.std(diff_similarities),
                'min': np.min(diff_similarities),
                'max': np.max(diff_similarities),
                'count': len(diff_similarities)
            },
            'mean_difference': np.mean(same_similarities) - np.mean(diff_similarities),
            'classification_metrics': {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1_score
            },
            'confusion_matrix': {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        }
    
    def test_multiple_thresholds(self, same_weapon_results: list, different_weapon_results: list):
        """Probar múltiples umbrales para encontrar el óptimo"""
        thresholds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        threshold_results = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(same_weapon_results, different_weapon_results, threshold)
            threshold_results.append(metrics)
            
            print(f"\n=== MÉTRICAS DE RENDIMIENTO (Umbral: {threshold}%) ===")
            print(f"Mismo arma - Media: {metrics['same_weapon_stats']['mean']:.2f}%, Std: {metrics['same_weapon_stats']['std']:.2f}%")
            print(f"Diferentes armas - Media: {metrics['different_weapon_stats']['mean']:.2f}%, Std: {metrics['different_weapon_stats']['std']:.2f}%")
            print(f"Diferencia de medias: {metrics['mean_difference']:.2f}%")
            print(f"\nMétricas de clasificación:")
            print(f"  Precision: {metrics['classification_metrics']['precision']:.3f}")
            print(f"  Recall: {metrics['classification_metrics']['recall']:.3f}")
            print(f"  Accuracy: {metrics['classification_metrics']['accuracy']:.3f}")
            print(f"  F1-Score: {metrics['classification_metrics']['f1_score']:.3f}")
            print(f"\nMatriz de confusión:")
            cm = metrics['confusion_matrix']
            print(f"  TP: {cm['tp']}, FP: {cm['fp']}")
            print(f"  FN: {cm['fn']}, TN: {cm['tn']}")
        
        # Encontrar mejor umbral basado en F1-Score
        best_threshold = max(threshold_results, key=lambda x: x['classification_metrics']['f1_score'])
        print(f"\nMejor umbral: {best_threshold['threshold']}% (F1-Score: {best_threshold['classification_metrics']['f1_score']:.3f})")
        
        return threshold_results, best_threshold

def main():
    tester = ImprovedMatchingTester()
    
    print("================================================================================")
    print("PRUEBAS DE MATCHING MEJORADO")
    print("================================================================================")
    
    # Descubrir imágenes
    weapon_images = tester.discover_images()
    
    if not weapon_images:
        print("ERROR: No se encontraron imágenes en el dataset")
        return
    
    print(f"\nArmas encontradas: {len(weapon_images)}")
    for weapon_id, images in weapon_images.items():
        print(f"  {weapon_id}: {len(images)} imágenes")
    
    # Pruebas de matching
    same_weapon_results = tester.test_same_weapon_matching(weapon_images, max_comparisons=30)
    different_weapon_results = tester.test_different_weapon_matching(weapon_images, max_comparisons=30)
    
    print(f"\nComparaciones realizadas:")
    print(f"  Mismo arma: {len(same_weapon_results)}")
    print(f"  Diferentes armas: {len(different_weapon_results)}")
    
    # Análisis de múltiples umbrales
    threshold_results, best_threshold = tester.test_multiple_thresholds(same_weapon_results, different_weapon_results)
    
    print("\n" + "="*80)
    print("MÉTRICAS FINALES CON UMBRAL ÓPTIMO")
    print("="*80)
    
    final_metrics = tester.calculate_metrics(
        same_weapon_results, 
        different_weapon_results, 
        best_threshold['threshold']
    )
    
    print(f"\n=== MÉTRICAS DE RENDIMIENTO (Umbral: {final_metrics['threshold']}%) ===")
    print(f"Mismo arma - Media: {final_metrics['same_weapon_stats']['mean']:.2f}%, Std: {final_metrics['same_weapon_stats']['std']:.2f}%")
    print(f"Diferentes armas - Media: {final_metrics['different_weapon_stats']['mean']:.2f}%, Std: {final_metrics['different_weapon_stats']['std']:.2f}%")
    print(f"Diferencia de medias: {final_metrics['mean_difference']:.2f}%")
    print(f"\nMétricas de clasificación:")
    print(f"  Precision: {final_metrics['classification_metrics']['precision']:.3f}")
    print(f"  Recall: {final_metrics['classification_metrics']['recall']:.3f}")
    print(f"  Accuracy: {final_metrics['classification_metrics']['accuracy']:.3f}")
    print(f"  F1-Score: {final_metrics['classification_metrics']['f1_score']:.3f}")
    print(f"\nMatriz de confusión:")
    cm = final_metrics['confusion_matrix']
    print(f"  TP: {cm['tp']}, FP: {cm['fp']}")
    print(f"  FN: {cm['fn']}, TN: {cm['tn']}")
    
    # Guardar resultados
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'matcher_type': 'ImprovedBallisticMatcher',
        'same_weapon_results': same_weapon_results,
        'different_weapon_results': different_weapon_results,
        'threshold_analysis': threshold_results,
        'best_threshold': best_threshold,
        'final_metrics': final_metrics
    }
    
    output_path = Path("reports/improved_matching_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    print("\n" + "="*80)
    print("PRUEBAS COMPLETADAS")
    print("="*80)

if __name__ == "__main__":
    main()