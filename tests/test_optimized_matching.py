#!/usr/bin/env python3
"""
Script de prueba para el matcher optimizado
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

class OptimizedMatchingTester:
    def __init__(self, dataset_path: str = "uploads/data"):
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__)
        self.matcher = UnifiedMatcher()
        
    def discover_images(self):
        """Descubrir imágenes agrupadas por arma"""
        weapon_images = defaultdict(list)
        
        for image_path in self.dataset_path.rglob("*.png"):
            weapon_id = self._get_weapon_id_from_filename(image_path.name)
            if weapon_id:
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
        print(f"\n=== PRUEBA MISMO ARMA (Matcher Optimizado) ===")
        
        same_weapon_results = []
        comparisons_done = 0
        
        for weapon_id, images in weapon_images.items():
            if comparisons_done >= max_comparisons:
                break
                
            print(f"\nProbando arma: {weapon_id} ({len(images)} imágenes)")
            
            # Comparar primeras imágenes de cada arma
            for i in range(min(3, len(images))):
                for j in range(i + 1, min(5, len(images))):
                    if comparisons_done >= max_comparisons:
                        break
                    
                    result = self.matcher.compare_image_files(images[i], images[j])
                    result_dict = {
                        'similarity': result.similarity_score,
                        'confidence': result.confidence,
                        'total_matches': result.total_matches,
                        'good_matches': result.good_matches,
                        'processing_time': result.processing_time,
                        'weapon_id': weapon_id,
                        'comparison_type': 'same_weapon'
                    }
                    
                    same_weapon_results.append(result_dict)
                    comparisons_done += 1
                    
                    print(f"  {images[i].name} vs {images[j].name}: {result.similarity_score:.2f}%")
        
        return same_weapon_results
    
    def test_different_weapon_matching(self, weapon_images: dict, max_comparisons: int = 50):
        """Probar matching entre imágenes de diferentes armas"""
        print(f"\n=== PRUEBA DIFERENTES ARMAS (Matcher Optimizado) ===")
        
        different_weapon_results = []
        comparisons_done = 0
        
        weapons = list(weapon_images.keys())
        
        for i, weapon1 in enumerate(weapons):
            if comparisons_done >= max_comparisons:
                break
                
            for j, weapon2 in enumerate(weapons[i+1:], i+1):
                if comparisons_done >= max_comparisons:
                    break
                
                # Comparar primera imagen de cada arma
                if weapon_images[weapon1] and weapon_images[weapon2]:
                    result = self.matcher.compare_image_files(
                        weapon_images[weapon1][0], 
                        weapon_images[weapon2][0]
                    )
                    result_dict = {
                        'similarity': result.similarity_score,
                        'confidence': result.confidence,
                        'total_matches': result.total_matches,
                        'good_matches': result.good_matches,
                        'processing_time': result.processing_time,
                        'weapon1_id': weapon1,
                        'weapon2_id': weapon2,
                        'comparison_type': 'different_weapon'
                    }
                    
                    different_weapon_results.append(result_dict)
                    comparisons_done += 1
                    
                    print(f"  {weapon1} vs {weapon2}: {result.similarity_score:.2f}%")
        
        return different_weapon_results
    
    def calculate_metrics(self, same_weapon_results: list, different_weapon_results: list, threshold: float = 20.0):
        """Calcular métricas de rendimiento"""
        print(f"\n=== MÉTRICAS DE RENDIMIENTO (Umbral: {threshold}%) ===")
        
        # Extraer similitudes
        same_similarities = [r['similarity'] for r in same_weapon_results if 'error' not in r]
        diff_similarities = [r['similarity'] for r in different_weapon_results if 'error' not in r]
        
        if not same_similarities or not diff_similarities:
            print("ERROR: No hay suficientes resultados para calcular métricas")
            return
        
        # Estadísticas básicas
        same_mean = np.mean(same_similarities)
        same_std = np.std(same_similarities)
        diff_mean = np.mean(diff_similarities)
        diff_std = np.std(diff_similarities)
        
        print(f"Mismo arma - Media: {same_mean:.2f}%, Std: {same_std:.2f}%")
        print(f"Diferentes armas - Media: {diff_mean:.2f}%, Std: {diff_std:.2f}%")
        print(f"Diferencia de medias: {same_mean - diff_mean:.2f}%")
        
        # Métricas de clasificación
        tp = sum(1 for s in same_similarities if s >= threshold)  # True Positives
        fn = sum(1 for s in same_similarities if s < threshold)   # False Negatives
        tn = sum(1 for s in diff_similarities if s < threshold)  # True Negatives
        fp = sum(1 for s in diff_similarities if s >= threshold) # False Positives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMétricas de clasificación:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
        
        print(f"\nMatriz de confusión:")
        print(f"  TP: {tp}, FP: {fp}")
        print(f"  FN: {fn}, TN: {tn}")
        
        return {
            'threshold': threshold,
            'same_weapon_stats': {
                'mean': same_mean,
                'std': same_std,
                'count': len(same_similarities)
            },
            'different_weapon_stats': {
                'mean': diff_mean,
                'std': diff_std,
                'count': len(diff_similarities)
            },
            'classification_metrics': {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        }
    
    def test_multiple_thresholds(self, same_weapon_results: list, different_weapon_results: list):
        """Probar múltiples umbrales para encontrar el óptimo"""
        print(f"\n=== ANÁLISIS DE UMBRALES ===")
        
        thresholds = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(same_weapon_results, different_weapon_results, threshold)
            if metrics:
                f1 = metrics['classification_metrics']['f1_score']
                print(f"Umbral {threshold}%: F1-Score = {f1:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        print(f"\nMejor umbral: {best_threshold}% (F1-Score: {best_f1:.3f})")
        return best_threshold, best_f1

def main():
    """Función principal"""
    tester = OptimizedMatchingTester()
    
    print("Iniciando pruebas del matcher optimizado...")
    
    # Descubrir imágenes
    weapon_images = tester.discover_images()
    
    if len(weapon_images) < 2:
        print("ERROR: Se necesitan al menos 2 armas para las pruebas")
        return
    
    print(f"Encontradas {len(weapon_images)} armas:")
    for weapon_id, images in weapon_images.items():
        print(f"  {weapon_id}: {len(images)} imágenes")
    
    # Pruebas de matching
    start_time = time.time()
    
    same_weapon_results = tester.test_same_weapon_matching(weapon_images, max_comparisons=30)
    different_weapon_results = tester.test_different_weapon_matching(weapon_images, max_comparisons=30)
    
    end_time = time.time()
    
    print(f"\nTiempo total de pruebas: {end_time - start_time:.2f} segundos")
    print(f"Comparaciones mismo arma: {len(same_weapon_results)}")
    print(f"Comparaciones diferentes armas: {len(different_weapon_results)}")
    
    # Calcular métricas
    if same_weapon_results and different_weapon_results:
        # Probar múltiples umbrales
        best_threshold, best_f1 = tester.test_multiple_thresholds(same_weapon_results, different_weapon_results)
        
        # Métricas detalladas con mejor umbral
        print(f"\n" + "="*80)
        print("MÉTRICAS FINALES CON UMBRAL ÓPTIMO")
        print("="*80)
        final_metrics = tester.calculate_metrics(same_weapon_results, different_weapon_results, best_threshold)
        
        # Guardar resultados
        results = {
            'timestamp': time.time(),
            'algorithm': 'OptimizedBallisticMatcher',
            'same_weapon_results': same_weapon_results,
            'different_weapon_results': different_weapon_results,
            'final_metrics': final_metrics,
            'best_threshold': best_threshold,
            'best_f1_score': best_f1
        }
        
        output_file = Path("reports/optimized_matching_results.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {output_file}")
    
    print("\n" + "="*80)
    print("PRUEBAS COMPLETADAS")
    print("="*80)

if __name__ == "__main__":
    main()