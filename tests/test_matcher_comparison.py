#!/usr/bin/env python3
"""
Script de comparación entre diferentes matchers
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

class MatcherComparison:
    def __init__(self, dataset_path: str = "uploads/data"):
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__)
        
        # Inicializar el matcher unificado
        self.matcher = UnifiedMatcher()
        
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
    
    def extract_features_for_basic_matcher(self, image_path: str):
        """Extraer características para el matcher básico"""
        # Cargar imagen
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Usar ORB (mismo que los otros matchers)
        orb = cv2.ORB_create(nfeatures=1000)
        kp, desc = orb.detectAndCompute(img, None)
        
        if desc is None:
            return None
        
        return {
            'keypoints': kp,
            'descriptors': desc,
            'algorithm': 'ORB'
        }
    
    def compare_single_pair(self, img1_path: str, img2_path: str):
        """Comparar un par de imágenes con el matcher unificado"""
        print(f"\n=== COMPARANDO PAR DE IMÁGENES ===")
        print(f"Imagen 1: {Path(img1_path).name}")
        print(f"Imagen 2: {Path(img2_path).name}")
        
        results = {}
        
        # Matcher Unificado
        try:
            result = self.matcher.compare_image_files(img1_path, img2_path)
            results['unified'] = {
                'similarity': result.similarity_score,
                'good_matches': result.good_matches,
                'total_matches': result.total_matches,
                'confidence': result.confidence,
                'processing_time': result.processing_time
            }
            print(f"Matcher Unificado: {result.similarity_score:.2f}% ({result.good_matches} matches)")
        except Exception as e:
            print(f"Matcher Unificado: ERROR - {e}")
            results['unified'] = None
        
        return results
    
    def analyze_differences(self, results: dict):
        """Analizar los resultados del matcher unificado"""
        print(f"\n=== ANÁLISIS DE RESULTADOS ===")
        
        if results['unified']:
            unified_sim = results['unified']['similarity']
            print(f"Matcher Unificado: {unified_sim:.2f}%")
            print(f"  - Good matches: {results['unified']['good_matches']}")
            print(f"  - Total matches: {results['unified']['total_matches']}")
            print(f"  - Confidence: {results['unified']['confidence']:.3f}")
            print(f"  - Processing time: {results['unified']['processing_time']:.3f}s")
        else:
            print("No se pudieron obtener resultados del matcher unificado")
    
    def run_comparison_test(self, max_pairs: int = 10):
        """Ejecutar pruebas de comparación"""
        print("================================================================================")
        print("COMPARACIÓN DE MATCHERS")
        print("================================================================================")
        
        # Descubrir imágenes
        weapon_images = self.discover_images()
        
        if not weapon_images:
            print("ERROR: No se encontraron imágenes en el dataset")
            return
        
        print(f"\nArmas encontradas: {len(weapon_images)}")
        for weapon_id, images in weapon_images.items():
            print(f"  {weapon_id}: {len(images)} imágenes")
        
        # Seleccionar pares para comparar
        comparison_results = []
        pairs_tested = 0
        
        # Probar pares del mismo arma
        print(f"\n=== PROBANDO PARES DEL MISMO ARMA ===")
        for weapon_id, images in weapon_images.items():
            if len(images) >= 2 and pairs_tested < max_pairs // 2:
                img1_path = str(images[0])
                img2_path = str(images[1])
                
                results = self.compare_single_pair(img1_path, img2_path)
                results['comparison_type'] = 'same_weapon'
                results['weapon_id'] = weapon_id
                results['image1_name'] = Path(img1_path).name
                results['image2_name'] = Path(img2_path).name
                
                self.analyze_differences(results)
                comparison_results.append(results)
                pairs_tested += 1
        
        # Probar pares de diferentes armas
        print(f"\n=== PROBANDO PARES DE DIFERENTES ARMAS ===")
        weapon_list = list(weapon_images.keys())
        for i in range(len(weapon_list)):
            for j in range(i + 1, len(weapon_list)):
                if pairs_tested >= max_pairs:
                    break
                
                weapon1 = weapon_list[i]
                weapon2 = weapon_list[j]
                
                if len(weapon_images[weapon1]) > 0 and len(weapon_images[weapon2]) > 0:
                    img1_path = str(weapon_images[weapon1][0])
                    img2_path = str(weapon_images[weapon2][0])
                    
                    results = self.compare_single_pair(img1_path, img2_path)
                    results['comparison_type'] = 'different_weapon'
                    results['weapon1_id'] = weapon1
                    results['weapon2_id'] = weapon2
                    results['image1_name'] = Path(img1_path).name
                    results['image2_name'] = Path(img2_path).name
                    
                    self.analyze_differences(results)
                    comparison_results.append(results)
                    pairs_tested += 1
            
            if pairs_tested >= max_pairs:
                break
        
        # Análisis estadístico
        self.statistical_analysis(comparison_results)
        
        # Guardar resultados
        output_path = Path("reports/matcher_comparison_results.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_comparisons': len(comparison_results),
                'results': comparison_results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nResultados guardados en: {output_path}")
    
    def statistical_analysis(self, results: list):
        """Análisis estadístico de los resultados"""
        print(f"\n" + "="*80)
        print("ANÁLISIS ESTADÍSTICO")
        print("="*80)
        
        # Separar por tipo de comparación
        same_weapon = [r for r in results if r.get('comparison_type') == 'same_weapon']
        diff_weapon = [r for r in results if r.get('comparison_type') == 'different_weapon']
        
        print(f"\nComparaciones mismo arma: {len(same_weapon)}")
        print(f"Comparaciones diferentes armas: {len(diff_weapon)}")
        
        # Análisis por matcher
        matchers = ['basic', 'optimized', 'improved']
        
        for matcher in matchers:
            print(f"\n--- {matcher.upper()} MATCHER ---")
            
            # Same weapon stats
            same_sims = []
            for r in same_weapon:
                if r.get(matcher) and r[matcher].get('similarity') is not None:
                    if matcher == 'basic':
                        same_sims.append(r[matcher]['similarity'])
                    else:
                        same_sims.append(r[matcher]['similarity'])
            
            # Different weapon stats
            diff_sims = []
            for r in diff_weapon:
                if r.get(matcher) and r[matcher].get('similarity') is not None:
                    if matcher == 'basic':
                        diff_sims.append(r[matcher]['similarity'])
                    else:
                        diff_sims.append(r[matcher]['similarity'])
            
            if same_sims:
                print(f"Mismo arma - Media: {np.mean(same_sims):.2f}%, Std: {np.std(same_sims):.2f}%")
                print(f"             Min: {np.min(same_sims):.2f}%, Max: {np.max(same_sims):.2f}%")
            
            if diff_sims:
                print(f"Diferentes armas - Media: {np.mean(diff_sims):.2f}%, Std: {np.std(diff_sims):.2f}%")
                print(f"                   Min: {np.min(diff_sims):.2f}%, Max: {np.max(diff_sims):.2f}%")
            
            if same_sims and diff_sims:
                difference = np.mean(same_sims) - np.mean(diff_sims)
                print(f"Diferencia de medias: {difference:.2f}%")

def main():
    comparison = MatcherComparison()
    comparison.run_comparison_test(max_pairs=10)
    
    print("\n" + "="*80)
    print("COMPARACIÓN COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()