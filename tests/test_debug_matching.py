#!/usr/bin/env python3
"""
Script de debug para identificar problemas en el matching de características
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from image_processing.feature_extractor import extract_orb_features
from matching.unified_matcher import UnifiedMatcher
from config.unified_config import get_unified_config
from utils.logger import get_logger

def debug_single_image_features(image_path: Path):
    """Debug de extracción de características en una sola imagen"""
    print(f"\n=== DEBUG IMAGEN: {image_path.name} ===")
    
    # Cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: No se pudo cargar la imagen {image_path}")
        return None
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Imagen cargada: {gray.shape}")
    
    # Extraer características
    features = extract_orb_features(gray, max_features=200)
    
    print(f"Características extraídas:")
    print(f"  - Keypoints: {features.get('num_keypoints', 0)}")
    print(f"  - Descriptors shape: {features.get('descriptors').shape if features.get('descriptors') is not None else 'None'}")
    print(f"  - Keypoints type: {type(features.get('keypoints'))}")
    print(f"  - Descriptors type: {type(features.get('descriptors'))}")
    
    return features

def debug_matching_process(img1_path: Path, img2_path: Path):
    """Debug del proceso completo de matching"""
    print(f"\n=== DEBUG MATCHING: {img1_path.name} vs {img2_path.name} ===")
    
    # Extraer características de ambas imágenes
    features1 = debug_single_image_features(img1_path)
    features2 = debug_single_image_features(img2_path)
    
    if not features1 or not features2:
        print("ERROR: No se pudieron extraer características")
        return
    
    if features1.get('num_keypoints', 0) == 0 or features2.get('num_keypoints', 0) == 0:
        print("ERROR: Una de las imágenes no tiene keypoints")
        return
    
    # Inicializar matcher
    matcher = UnifiedMatcher()
    
    print(f"\nRealizando matching...")
    print(f"Features1 keys: {list(features1.keys())}")
    print(f"Features2 keys: {list(features2.keys())}")
    
    # Realizar matching
    try:
        result = matcher.match_features(features1, features2)
        
        print(f"\nResultado del matching:")
        print(f"  - Similarity score: {result.similarity_score}%")
        print(f"  - Total matches: {result.total_matches}")
        print(f"  - Good matches: {result.good_matches}")
        print(f"  - Confidence: {result.confidence}")
        print(f"  - Processing time: {result.processing_time:.3f}s")
        
        return result
        
    except Exception as e:
        print(f"ERROR en matching: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_orb_matcher_directly(features1, features2):
    """Debug directo del matcher ORB"""
    print(f"\n=== DEBUG MATCHER ORB DIRECTO ===")
    
    desc1 = features1.get('descriptors')
    desc2 = features2.get('descriptors')
    kp1 = features1.get('keypoints', [])
    kp2 = features2.get('keypoints', [])
    
    print(f"Descriptors 1: {desc1.shape if desc1 is not None else 'None'}")
    print(f"Descriptors 2: {desc2.shape if desc2 is not None else 'None'}")
    print(f"Keypoints 1: {len(kp1)}")
    print(f"Keypoints 2: {len(kp2)}")
    
    if desc1 is None or desc2 is None:
        print("ERROR: Descriptors son None")
        return
    
    # Crear matcher BF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    try:
        matches = bf.match(desc1, desc2)
        print(f"Matches encontrados: {len(matches)}")
        
        if len(matches) > 0:
            # Ordenar por distancia
            matches = sorted(matches, key=lambda x: x.distance)
            
            distances = [m.distance for m in matches]
            print(f"Distancias - Min: {min(distances)}, Max: {max(distances)}, Mean: {np.mean(distances):.2f}")
            
            # Aplicar threshold simple
            good_matches = [m for m in matches if m.distance < 50]  # Threshold fijo para debug
            print(f"Good matches (dist < 50): {len(good_matches)}")
            
        return matches
        
    except Exception as e:
        print(f"ERROR en BF matcher: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal de debug"""
    logger = get_logger(__name__)
    
    # Buscar imágenes de prueba
    dataset_path = Path("uploads/data")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset no encontrado en {dataset_path}")
        return
    
    # Buscar algunas imágenes
    image_files = list(dataset_path.rglob("*.png"))[:4]  # Tomar solo 4 imágenes
    
    if len(image_files) < 2:
        print(f"ERROR: Se necesitan al menos 2 imágenes. Encontradas: {len(image_files)}")
        return
    
    print(f"Imágenes encontradas: {len(image_files)}")
    for img in image_files:
        print(f"  - {img.name}")
    
    # Test 1: Debug de características individuales
    print("\n" + "="*60)
    print("TEST 1: EXTRACCIÓN DE CARACTERÍSTICAS INDIVIDUALES")
    print("="*60)
    
    for img_path in image_files[:2]:
        debug_single_image_features(img_path)
    
    # Test 2: Debug de matching entre dos imágenes
    print("\n" + "="*60)
    print("TEST 2: MATCHING ENTRE DOS IMÁGENES")
    print("="*60)
    
    result = debug_matching_process(image_files[0], image_files[1])
    
    # Test 3: Debug directo del matcher
    if result is None:
        print("\n" + "="*60)
        print("TEST 3: DEBUG DIRECTO DEL MATCHER")
        print("="*60)
        
        img1 = cv2.imread(str(image_files[0]))
        img2 = cv2.imread(str(image_files[1]))
        
        if img1 is not None and img2 is not None:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            features1 = extract_orb_features(gray1, max_features=200)
            features2 = extract_orb_features(gray2, max_features=200)
            
            debug_orb_matcher_directly(features1, features2)

if __name__ == "__main__":
    main()