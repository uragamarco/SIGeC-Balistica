"""
Ejemplo de Integración - Procesamiento Paralelo de ROIs
Sistema Balístico Forense SEACABA

Este ejemplo demuestra cómo integrar el procesamiento paralelo
en aplicaciones reales del sistema balístico forense.
"""

import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from image_processing.ballistic_features_parallel import (
    ParallelBallisticFeatureExtractor,
    ParallelConfig,
    extract_ballistic_features_parallel
)
from image_processing.ballistic_features import BallisticFeatureExtractor

def create_sample_images():
    """Crea imágenes de muestra para demostración"""
    images = {}
    
    # Imagen de casquillo (cartridge case)
    cartridge = np.random.randint(50, 200, (1024, 1024), dtype=np.uint8)
    # Simular marca de percutor
    cv2.circle(cartridge, (512, 512), 40, 255, -1)
    cv2.circle(cartridge, (512, 512), 35, 100, -1)
    # Simular textura de culata
    for i in range(0, 1024, 15):
        cv2.line(cartridge, (i, 0), (i, 1024), 180, 1)
    images['cartridge_case'] = cartridge
    
    # Imagen de bala (bullet)
    bullet = np.random.randint(80, 180, (800, 800), dtype=np.uint8)
    # Simular estrías
    for i in range(0, 800, 20):
        cv2.line(bullet, (0, i), (800, i), 220, 2)
    # Agregar algunas marcas circulares
    cv2.circle(bullet, (400, 200), 25, 255, 2)
    cv2.circle(bullet, (400, 600), 30, 255, 2)
    images['bullet'] = bullet
    
    return images

def example_basic_usage():
    """Ejemplo básico de uso del procesamiento paralelo"""
    print("=== EJEMPLO 1: Uso Básico ===")
    
    # Crear imagen de muestra
    images = create_sample_images()
    cartridge_image = images['cartridge_case']
    
    # Método 1: Función de conveniencia (Recomendado)
    print("Procesando con función de conveniencia...")
    start_time = time.time()
    
    result = extract_ballistic_features_parallel(cartridge_image, 'cartridge_case')
    
    processing_time = time.time() - start_time
    
    if result['processing_success']:
        features = result['features']
        performance = result['performance']
        
        print(f"✅ Procesamiento exitoso en {processing_time:.2f}s")
        print(f"   Calidad: {features.quality_score:.3f}")
        print(f"   Confianza: {features.confidence:.3f}")
        print(f"   Memoria utilizada: {performance['memory_usage_mb']:.1f}MB")
        print(f"   Speedup: {performance['speedup_factor']:.2f}x")
    else:
        print(f"❌ Error en procesamiento: {result['error']}")
    
    print()

def example_advanced_configuration():
    """Ejemplo con configuración avanzada"""
    print("=== EJEMPLO 2: Configuración Avanzada ===")
    
    images = create_sample_images()
    bullet_image = images['bullet']
    
    # Configuración personalizada
    config = ParallelConfig(
        max_workers_process=4,      # 4 procesos para CPU intensivo
        max_workers_thread=8,       # 8 threads para I/O
        enable_gabor_parallel=True, # Habilitar paralelización Gabor
        memory_limit_gb=3.0,        # Límite de memoria 3GB
        chunk_size=6                # Procesar en chunks de 6
    )
    
    print(f"Configuración personalizada:")
    print(f"  - Procesos: {config.max_workers_process}")
    print(f"  - Threads: {config.max_workers_thread}")
    print(f"  - Gabor paralelo: {config.enable_gabor_parallel}")
    print(f"  - Límite memoria: {config.memory_limit_gb}GB")
    
    # Crear extractor con configuración personalizada
    extractor = ParallelBallisticFeatureExtractor(config)
    
    # Procesar imagen
    start_time = time.time()
    features = extractor.extract_ballistic_features_parallel(bullet_image, 'bullet')
    processing_time = time.time() - start_time
    
    # Obtener estadísticas detalladas
    stats = extractor.get_performance_stats()
    
    print(f"\n✅ Procesamiento completado en {processing_time:.2f}s")
    print(f"   Características de bala:")
    print(f"   - Densidad de estrías: {features.striation_density:.3f}")
    print(f"   - Orientación de estrías: {features.striation_orientation:.1f}°")
    print(f"   - Amplitud de estrías: {features.striation_amplitude:.3f}")
    print(f"   - Frecuencia de estrías: {features.striation_frequency:.3f}")
    print(f"   - Rugosidad de culata: {features.breech_face_roughness:.3f}")
    print(f"   - Diámetro de percutor: {features.firing_pin_diameter:.2f}")
    
    print(f"\n   Estadísticas de rendimiento:")
    print(f"   - Tiempo paralelo: {stats['parallel_time']:.2f}s")
    print(f"   - Memoria pico: {stats['memory_usage_mb']:.1f}MB")
    print(f"   - Factor speedup: {stats['speedup_factor']:.2f}x")
    
    print()

def example_comparison():
    """Ejemplo comparando procesamiento secuencial vs paralelo"""
    print("=== EJEMPLO 3: Comparación Secuencial vs Paralelo ===")
    
    images = create_sample_images()
    test_image = images['cartridge_case']
    
    # Procesamiento secuencial
    print("Procesando secuencialmente...")
    sequential_extractor = BallisticFeatureExtractor()
    
    start_time = time.time()
    seq_features = sequential_extractor.extract_ballistic_features(test_image, 'cartridge_case')
    seq_time = time.time() - start_time
    
    # Procesamiento paralelo
    print("Procesando en paralelo...")
    parallel_extractor = ParallelBallisticFeatureExtractor()
    
    start_time = time.time()
    par_features = parallel_extractor.extract_ballistic_features_parallel(test_image, 'cartridge_case')
    par_time = time.time() - start_time
    
    # Comparar resultados
    speedup = seq_time / par_time if par_time > 0 else 1.0
    
    print(f"\n📊 Comparación de Resultados:")
    print(f"   Tiempo secuencial: {seq_time:.2f}s")
    print(f"   Tiempo paralelo:   {par_time:.2f}s")
    print(f"   Speedup:           {speedup:.2f}x")
    
    print(f"\n   Calidad secuencial: {seq_features.quality_score:.3f}")
    print(f"   Calidad paralela:   {par_features.quality_score:.3f}")
    print(f"   Diferencia:         {abs(seq_features.quality_score - par_features.quality_score):.3f}")
    
    # Verificar consistencia
    quality_diff = abs(seq_features.quality_score - par_features.quality_score)
    confidence_diff = abs(seq_features.confidence - par_features.confidence)
    
    if quality_diff < 0.1 and confidence_diff < 0.1:
        print("   ✅ Resultados consistentes entre métodos")
    else:
        print("   ⚠️  Diferencias significativas detectadas")
    
    print()

def example_batch_processing():
    """Ejemplo de procesamiento en lote"""
    print("=== EJEMPLO 4: Procesamiento en Lote ===")
    
    # Crear múltiples imágenes de prueba
    images = create_sample_images()
    test_cases = [
        ('cartridge_1', images['cartridge_case'], 'cartridge_case'),
        ('cartridge_2', images['cartridge_case'], 'cartridge_case'),
        ('bullet_1', images['bullet'], 'bullet'),
        ('bullet_2', images['bullet'], 'bullet'),
    ]
    
    # Configuración optimizada para lote
    config = ParallelConfig(
        max_workers_process=2,  # Menos procesos para evitar sobrecarga
        max_workers_thread=4,   # Threads moderados
        enable_gabor_parallel=True,
        memory_limit_gb=2.0
    )
    
    extractor = ParallelBallisticFeatureExtractor(config)
    results = []
    
    print(f"Procesando {len(test_cases)} imágenes en lote...")
    total_start = time.time()
    
    for name, image, specimen_type in test_cases:
        print(f"  Procesando {name}...")
        
        start_time = time.time()
        features = extractor.extract_ballistic_features_parallel(image, specimen_type)
        processing_time = time.time() - start_time
        
        results.append({
            'name': name,
            'type': specimen_type,
            'features': features,
            'time': processing_time
        })
        
        print(f"    ✅ Completado en {processing_time:.2f}s (calidad: {features.quality_score:.3f})")
    
    total_time = time.time() - total_start
    avg_time = total_time / len(test_cases)
    
    print(f"\n📈 Resumen del Lote:")
    print(f"   Total de imágenes: {len(test_cases)}")
    print(f"   Tiempo total: {total_time:.2f}s")
    print(f"   Tiempo promedio: {avg_time:.2f}s por imagen")
    
    # Estadísticas por tipo
    cartridge_results = [r for r in results if r['type'] == 'cartridge_case']
    bullet_results = [r for r in results if r['type'] == 'bullet']
    
    if cartridge_results:
        avg_cartridge_quality = np.mean([r['features'].quality_score for r in cartridge_results])
        print(f"   Calidad promedio casquillos: {avg_cartridge_quality:.3f}")
    
    if bullet_results:
        avg_bullet_quality = np.mean([r['features'].quality_score for r in bullet_results])
        print(f"   Calidad promedio balas: {avg_bullet_quality:.3f}")
    
    print()

def example_error_handling():
    """Ejemplo de manejo de errores y casos límite"""
    print("=== EJEMPLO 5: Manejo de Errores ===")
    
    # Caso 1: Imagen muy pequeña
    print("Caso 1: Imagen muy pequeña")
    small_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    
    result = extract_ballistic_features_parallel(small_image, 'cartridge_case')
    
    if result['processing_success']:
        print(f"   ✅ Procesado exitosamente (calidad: {result['features'].quality_score:.3f})")
    else:
        print(f"   ❌ Error: {result['error']}")
    
    # Caso 2: Imagen con poco contraste
    print("\nCaso 2: Imagen con poco contraste")
    low_contrast = np.full((400, 400), 128, dtype=np.uint8)  # Gris uniforme
    
    result = extract_ballistic_features_parallel(low_contrast, 'bullet')
    
    if result['processing_success']:
        print(f"   ✅ Procesado (calidad baja esperada: {result['features'].quality_score:.3f})")
    else:
        print(f"   ❌ Error: {result['error']}")
    
    # Caso 3: Configuración con recursos limitados
    print("\nCaso 3: Configuración con recursos muy limitados")
    limited_config = ParallelConfig(
        max_workers_process=1,
        max_workers_thread=1,
        enable_gabor_parallel=False,
        memory_limit_gb=0.5
    )
    
    try:
        extractor = ParallelBallisticFeatureExtractor(limited_config)
        test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        features = extractor.extract_ballistic_features_parallel(test_image, 'cartridge_case')
        print(f"   ✅ Procesado con recursos limitados (calidad: {features.quality_score:.3f})")
    except Exception as e:
        print(f"   ❌ Error con recursos limitados: {e}")
    
    print()

def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🔬 EJEMPLOS DE PROCESAMIENTO PARALELO - SISTEMA BALÍSTICO FORENSE")
    print("=" * 70)
    
    try:
        # Ejecutar todos los ejemplos
        example_basic_usage()
        example_advanced_configuration()
        example_comparison()
        example_batch_processing()
        example_error_handling()
        
        print("🎉 TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("\nPara usar el procesamiento paralelo en tu aplicación:")
        print("1. Importa: from image_processing.ballistic_features_parallel import extract_ballistic_features_parallel")
        print("2. Usa: result = extract_ballistic_features_parallel(image, specimen_type)")
        print("3. Verifica: if result['processing_success']: ...")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución de ejemplos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()