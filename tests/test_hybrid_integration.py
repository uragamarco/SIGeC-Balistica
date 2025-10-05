#!/usr/bin/env python3
"""
Script de prueba para verificar la integración híbrida de ballistic_features.py
Prueba las nuevas funcionalidades de análisis de estrías mejoradas
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Agregar el directorio actual al path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from image_processing.ballistic_features import BallisticFeatureExtractor, BallisticFeatures, ParallelConfig
    print("✓ Importación exitosa de ballistic_features")
except ImportError as e:
    print(f"✗ Error al importar ballistic_features: {e}")
    sys.exit(1)

def create_synthetic_cartridge_image(width=800, height=600):
    """Crea una imagen sintética de casquillo para pruebas"""
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Fondo con ruido
    noise = np.random.normal(50, 15, (height, width))
    image = np.clip(noise, 0, 255).astype(np.uint8)
    
    # Círculo central (percutor)
    center = (width // 2, height // 2)
    cv2.circle(image, center, 30, 200, -1)
    cv2.circle(image, center, 25, 150, 2)
    
    # Estrías radiales
    for angle in range(0, 360, 15):
        x1 = center[0] + int(50 * np.cos(np.radians(angle)))
        y1 = center[1] + int(50 * np.sin(np.radians(angle)))
        x2 = center[0] + int(150 * np.cos(np.radians(angle)))
        y2 = center[1] + int(150 * np.sin(np.radians(angle)))
        cv2.line(image, (x1, y1), (x2, y2), 180, 2)
    
    # Textura de culata
    for i in range(0, width, 20):
        for j in range(0, height, 20):
            if np.sqrt((i - center[0])**2 + (j - center[1])**2) > 200:
                cv2.rectangle(image, (i, j), (i+10, j+10), 120, -1)
    
    return image

def test_basic_functionality():
    """Prueba la funcionalidad básica del extractor"""
    print("\n=== Prueba de Funcionalidad Básica ===")
    
    try:
        # Crear extractor
        extractor = BallisticFeatureExtractor()
        print("✓ Extractor creado exitosamente")
        
        # Crear imagen de prueba
        test_image = create_synthetic_cartridge_image()
        print("✓ Imagen sintética creada")
        
        # Extraer características
        features = extractor.extract_ballistic_features(test_image, 'cartridge_case')
        print("✓ Características extraídas exitosamente")
        
        # Verificar estructura
        assert isinstance(features, BallisticFeatures), "El resultado debe ser BallisticFeatures"
        print("✓ Estructura de datos correcta")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en prueba básica: {e}")
        return False

def test_new_striation_features():
    """Prueba las nuevas características de estrías"""
    print("\n=== Prueba de Nuevas Características de Estrías ===")
    
    try:
        extractor = BallisticFeatureExtractor()
        test_image = create_synthetic_cartridge_image()
        
        features = extractor.extract_ballistic_features(test_image, 'cartridge_case')
        
        # Verificar nuevos campos
        assert hasattr(features, 'striation_num_lines'), "Debe tener striation_num_lines"
        assert hasattr(features, 'striation_dominant_directions'), "Debe tener striation_dominant_directions"
        assert hasattr(features, 'striation_parallelism_score'), "Debe tener striation_parallelism_score"
        
        print(f"✓ Número de líneas de estrías: {features.striation_num_lines}")
        print(f"✓ Direcciones dominantes: {len(features.striation_dominant_directions)} detectadas")
        print(f"✓ Score de paralelismo: {features.striation_parallelism_score:.3f}")
        
        # Verificar que los valores son razonables
        assert features.striation_num_lines >= 0, "Número de líneas debe ser no negativo"
        assert 0 <= features.striation_parallelism_score <= 1, "Score de paralelismo debe estar entre 0 y 1"
        assert isinstance(features.striation_dominant_directions, list), "Direcciones debe ser una lista"
        
        print("✓ Todas las nuevas características funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"✗ Error en prueba de nuevas características: {e}")
        return False

def test_parallel_processing():
    """Prueba el procesamiento paralelo"""
    print("\n=== Prueba de Procesamiento Paralelo ===")
    
    try:
        # Configuración paralela
        parallel_config = ParallelConfig(
            max_workers_process=2,
            max_workers_thread=2,
            enable_gabor_parallel=True,
            enable_roi_parallel=True
        )
        
        extractor = BallisticFeatureExtractor(parallel_config)
        test_image = create_synthetic_cartridge_image()
        
        # Prueba con paralelización
        features_parallel = extractor.extract_ballistic_features(test_image, 'cartridge_case', use_parallel=True)
        print("✓ Procesamiento paralelo exitoso")
        
        # Prueba sin paralelización
        features_sequential = extractor.extract_ballistic_features(test_image, 'cartridge_case', use_parallel=False)
        print("✓ Procesamiento secuencial exitoso")
        
        # Comparar resultados (deben ser similares)
        diff_density = abs(features_parallel.striation_density - features_sequential.striation_density)
        print(f"✓ Diferencia en densidad de estrías: {diff_density:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en prueba paralela: {e}")
        return False

def test_empty_features():
    """Prueba la creación de características vacías"""
    print("\n=== Prueba de Características Vacías ===")
    
    try:
        extractor = BallisticFeatureExtractor()
        empty_features = extractor._create_empty_features()
        
        # Verificar nuevos campos en características vacías
        assert empty_features.striation_num_lines == 0, "Líneas vacías debe ser 0"
        assert empty_features.striation_dominant_directions == [], "Direcciones vacías debe ser lista vacía"
        assert empty_features.striation_parallelism_score == 0.0, "Score vacío debe ser 0.0"
        
        print("✓ Características vacías creadas correctamente")
        return True
        
    except Exception as e:
        print(f"✗ Error en prueba de características vacías: {e}")
        return False

def test_compatibility():
    """Prueba la compatibilidad con el resto del sistema"""
    print("\n=== Prueba de Compatibilidad ===")
    
    try:
        extractor = BallisticFeatureExtractor()
        test_image = create_synthetic_cartridge_image()
        
        # Probar métodos individuales
        breech_features = extractor.extract_breech_face_features(test_image)
        firing_pin_features = extractor.extract_firing_pin_features(test_image)
        all_features = extractor.extract_all_features(test_image)
        
        print("✓ Métodos individuales funcionan")
        print(f"✓ Características de culata: {len(breech_features)} métricas")
        print(f"✓ Características de percutor: {len(firing_pin_features)} métricas")
        print(f"✓ Todas las características: {len(all_features)} categorías")
        
        return True
        
    except Exception as e:
        print(f"✗ Error en prueba de compatibilidad: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🔬 Iniciando pruebas de integración híbrida")
    print("=" * 50)
    
    tests = [
        ("Funcionalidad Básica", test_basic_functionality),
        ("Nuevas Características de Estrías", test_new_striation_features),
        ("Procesamiento Paralelo", test_parallel_processing),
        ("Características Vacías", test_empty_features),
        ("Compatibilidad", test_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Ejecutando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen de resultados
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        print(f"{status:<10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado final: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! La integración híbrida funciona correctamente.")
        return 0
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar la implementación.")
        return 1

if __name__ == "__main__":
    sys.exit(main())