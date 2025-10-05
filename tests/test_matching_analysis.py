#!/usr/bin/env python3
"""
Script de análisis detallado del matching para identificar problemas de similitud
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from collections import defaultdict

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from image_processing.feature_extractor import extract_orb_features, extract_sift_features
from matching.unified_matcher import UnifiedMatcher
from utils.logger import get_logger

class MatchingAnalyzer:
    def __init__(self, dataset_path: str = "uploads/data"):
        self.dataset_path = Path(dataset_path)
        self.logger = get_logger(__name__)
        self.matcher = UnifiedMatcher()
        
    def analyze_match_distances(self, img1_path: Path, img2_path: Path, algorithm="ORB"):
        """Analizar distribución de distancias en matches"""
        print(f"\n=== ANÁLISIS DE DISTANCIAS: {img1_path.name} vs {img2_path.name} ===")
        
        # Cargar imágenes
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            print("ERROR: No se pudieron cargar las imágenes")
            return None
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Extraer características
        if algorithm == "ORB":
            features1 = extract_orb_features(gray1, max_features=500)
            features2 = extract_orb_features(gray2, max_features=500)
        else:
            features1 = extract_sift_features(gray1, max_features=500)
            features2 = extract_sift_features(gray2, max_features=500)
        
        desc1 = features1.get('descriptors')
        desc2 = features2.get('descriptors')
        
        if desc1 is None or desc2 is None:
            print("ERROR: No se pudieron extraer descriptores")
            return None
        
        print(f"Descriptors 1: {desc1.shape}")
        print(f"Descriptors 2: {desc2.shape}")
        
        # Matching directo
        if algorithm == "ORB":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        distances = [m.distance for m in matches]
        
        print(f"Total matches: {len(matches)}")
        if len(distances) > 0:
            print(f"Distancia - Min: {min(distances)}, Max: {max(distances)}")
            print(f"Distancia - Mean: {np.mean(distances):.2f}, Std: {np.std(distances):.2f}")
            print(f"Distancia - Percentiles: 25%={np.percentile(distances, 25):.1f}, 50%={np.percentile(distances, 50):.1f}, 75%={np.percentile(distances, 75):.1f}")
        
        # Probar diferentes thresholds de Lowe's ratio
        print(f"\n--- Análisis de Lowe's Ratio ---")
        for ratio in [0.7, 0.75, 0.8, 0.85, 0.9]:
            good_matches = self._apply_lowes_ratio(desc1, desc2, ratio, algorithm)
            print(f"Ratio {ratio}: {len(good_matches)} good matches")
        
        return {
            'total_matches': len(matches),
            'distances': distances,
            'mean_distance': np.mean(distances) if distances else 0,
            'std_distance': np.std(distances) if distances else 0
        }
    
    def _apply_lowes_ratio(self, desc1, desc2, ratio, algorithm):
        """Aplicar Lowe's ratio test con ratio específico"""
        if algorithm == "ORB":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2)
        
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def analyze_by_ammunition_type(self):
        """Analizar matches agrupados por tipo de munición"""
        print(f"\n=== ANÁLISIS POR TIPO DE MUNICIÓN ===")
        
        # Descubrir imágenes
        weapon_images = self._discover_images()
        
        ammunition_analysis = defaultdict(list)
        
        # Analizar primer arma como ejemplo
        weapon_id = list(weapon_images.keys())[0]
        images = weapon_images[weapon_id][:10]  # Limitar para análisis
        
        print(f"Analizando arma: {weapon_id}")
        
        # Agrupar por tipo de munición
        ammo_groups = defaultdict(list)
        for img_path in images:
            # Extraer tipo de munición del nombre
            filename = img_path.name
            for ammo_type in ['CCI', 'FED', 'WIN', 'SPR', 'WOLF', 'RP']:
                if ammo_type in filename:
                    ammo_groups[ammo_type].append(img_path)
                    break
        
        print(f"Grupos de munición encontrados: {list(ammo_groups.keys())}")
        
        # Comparar dentro del mismo tipo de munición
        for ammo_type, ammo_images in ammo_groups.items():
            if len(ammo_images) >= 2:
                print(f"\n--- Mismo tipo de munición ({ammo_type}) ---")
                result = self.analyze_match_distances(ammo_images[0], ammo_images[1])
                if result:
                    ammunition_analysis[f'same_ammo_{ammo_type}'].append(result['mean_distance'])
        
        # Comparar entre diferentes tipos de munición
        ammo_types = list(ammo_groups.keys())
        if len(ammo_types) >= 2:
            print(f"\n--- Diferentes tipos de munición ({ammo_types[0]} vs {ammo_types[1]}) ---")
            if ammo_groups[ammo_types[0]] and ammo_groups[ammo_types[1]]:
                result = self.analyze_match_distances(ammo_groups[ammo_types[0]][0], ammo_groups[ammo_types[1]][0])
                if result:
                    ammunition_analysis['different_ammo'].append(result['mean_distance'])
        
        return ammunition_analysis
    
    def compare_orb_vs_sift(self, img1_path: Path, img2_path: Path):
        """Comparar rendimiento de ORB vs SIFT"""
        print(f"\n=== COMPARACIÓN ORB vs SIFT ===")
        print(f"Imágenes: {img1_path.name} vs {img2_path.name}")
        
        # Análisis con ORB
        print(f"\n--- ORB ---")
        orb_result = self.analyze_match_distances(img1_path, img2_path, "ORB")
        
        # Análisis con SIFT (saltear si hay problemas)
        print(f"\n--- SIFT ---")
        try:
            sift_result = self.analyze_match_distances(img1_path, img2_path, "SIFT")
        except Exception as e:
            print(f"ERROR con SIFT: {e}")
            sift_result = None
        
        # Comparación
        if orb_result and sift_result:
            print(f"\n--- COMPARACIÓN ---")
            print(f"ORB - Total matches: {orb_result['total_matches']}, Mean distance: {orb_result['mean_distance']:.2f}")
            print(f"SIFT - Total matches: {sift_result['total_matches']}, Mean distance: {sift_result['mean_distance']:.2f}")
        elif orb_result:
            print(f"\n--- SOLO ORB DISPONIBLE ---")
            print(f"ORB - Total matches: {orb_result['total_matches']}, Mean distance: {orb_result['mean_distance']:.2f}")
        
        return orb_result, sift_result
    
    def _discover_images(self):
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
    
    def test_optimized_matching(self, img1_path: Path, img2_path: Path):
        """Probar matching con parámetros optimizados"""
        print(f"\n=== TEST MATCHING OPTIMIZADO ===")
        print(f"Imágenes: {img1_path.name} vs {img2_path.name}")
        
        # Cargar imágenes
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Extraer características con más features
        features1 = extract_orb_features(gray1, max_features=1000)
        features2 = extract_orb_features(gray2, max_features=1000)
        
        desc1 = features1.get('descriptors')
        desc2 = features2.get('descriptors')
        kp1 = features1.get('keypoints')
        kp2 = features2.get('keypoints')
        
        if desc1 is None or desc2 is None:
            print("ERROR: No se pudieron extraer descriptores")
            return
        
        # Matching con Lowe's ratio optimizado
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Probar diferentes ratios
        for ratio in [0.75, 0.8, 0.85]:
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
            
            # Filtrado RANSAC si hay suficientes matches
            if len(good_matches) >= 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matches_mask = mask.ravel().tolist()
                    ransac_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
                    
                    # Calcular similitud mejorada
                    similarity = self._calculate_improved_similarity(len(ransac_matches), len(kp1), len(kp2))
                    
                    print(f"Ratio {ratio}: {len(good_matches)} good matches, {len(ransac_matches)} after RANSAC, Similarity: {similarity:.2f}%")
                    
                except:
                    similarity = len(good_matches) / max(len(kp1), len(kp2)) * 100
                    print(f"Ratio {ratio}: {len(good_matches)} good matches, RANSAC failed, Similarity: {similarity:.2f}%")
            else:
                similarity = len(good_matches) / max(len(kp1), len(kp2)) * 100
                print(f"Ratio {ratio}: {len(good_matches)} good matches, insufficient for RANSAC, Similarity: {similarity:.2f}%")
    
    def _calculate_improved_similarity(self, good_matches, kp1_count, kp2_count):
        """Calcular similitud mejorada"""
        if kp1_count == 0 or kp2_count == 0:
            return 0.0
        
        # Fórmula mejorada que considera la densidad de matches
        max_possible_matches = min(kp1_count, kp2_count)
        match_ratio = good_matches / max_possible_matches
        
        # Bonus por alta densidad de matches
        density_bonus = 1.0
        if good_matches > 20:
            density_bonus = 1.2
        elif good_matches > 50:
            density_bonus = 1.5
        
        similarity = match_ratio * 100 * density_bonus
        return min(similarity, 100.0)  # Cap at 100%

def main():
    """Función principal de análisis"""
    analyzer = MatchingAnalyzer()
    
    # Buscar imágenes de prueba
    weapon_images = analyzer._discover_images()
    
    if len(weapon_images) < 2:
        print("ERROR: Se necesitan al menos 2 armas para el análisis")
        return
    
    # Tomar primeras dos armas
    weapons = list(weapon_images.keys())[:2]
    weapon1_images = weapon_images[weapons[0]][:3]
    weapon2_images = weapon_images[weapons[1]][:3]
    
    print(f"Analizando armas: {weapons[0]} ({len(weapon1_images)} imágenes) vs {weapons[1]} ({len(weapon2_images)} imágenes)")
    
    # Test 1: Análisis de distancias - mismo arma
    print("\n" + "="*80)
    print("TEST 1: ANÁLISIS MISMO ARMA")
    print("="*80)
    analyzer.analyze_match_distances(weapon1_images[0], weapon1_images[1])
    
    # Test 2: Análisis de distancias - diferentes armas
    print("\n" + "="*80)
    print("TEST 2: ANÁLISIS DIFERENTES ARMAS")
    print("="*80)
    analyzer.analyze_match_distances(weapon1_images[0], weapon2_images[0])
    
    # Test 3: Comparación ORB vs SIFT
    print("\n" + "="*80)
    print("TEST 3: COMPARACIÓN ORB vs SIFT")
    print("="*80)
    analyzer.compare_orb_vs_sift(weapon1_images[0], weapon1_images[1])
    
    # Test 4: Análisis por tipo de munición
    print("\n" + "="*80)
    print("TEST 4: ANÁLISIS POR MUNICIÓN")
    print("="*80)
    analyzer.analyze_by_ammunition_type()
    
    # Test 5: Matching optimizado
    print("\n" + "="*80)
    print("TEST 5: MATCHING OPTIMIZADO")
    print("="*80)
    analyzer.test_optimized_matching(weapon1_images[0], weapon1_images[1])
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()