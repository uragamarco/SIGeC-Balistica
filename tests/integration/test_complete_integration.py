#!/usr/bin/env python3
"""
Script de pruebas de integración completa para el sistema SIGeC-Balistica
Valida el flujo completo desde preprocesamiento hasta extracción de características
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_images():
    """Crear imágenes de prueba para diferentes tipos de evidencia"""
    test_images = {}
    
    # Imagen de casquillo
    casquillo = np.zeros((800, 600, 3), dtype=np.uint8)
    # Simular marca de percutor (círculo central)
    cv2.circle(casquillo, (300, 400), 50, (200, 200, 200), -1)
    cv2.circle(casquillo, (300, 400), 30, (100, 100, 100), -1)
    # Simular marcas de extractor
    cv2.ellipse(casquillo, (300, 300), (80, 20), 45, 0, 360, (150, 150, 150), 3)
    # Simular cara de recámara
    cv2.rectangle(casquillo, (200, 200), (400, 600), (180, 180, 180), 2)
    test_images['casquillo'] = casquillo
    
    # Imagen de proyectil
    proyectil = np.zeros((800, 600, 3), dtype=np.uint8)
    # Simular estrías (líneas paralelas)
    for i in range(10, 590, 20):
        cv2.line(proyectil, (i, 100), (i, 700), (160, 160, 160), 2)
    # Agregar algo de ruido y variación
    noise = np.random.normal(0, 10, proyectil.shape).astype(np.uint8)
    proyectil = cv2.add(proyectil, noise)
    test_images['proyectil'] = proyectil
    
    return test_images

def test_preprocessing():
    """Probar el módulo de preprocesamiento"""
    print("=== PROBANDO PREPROCESAMIENTO ===")
    
    try:
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        
        preprocessor = UnifiedPreprocessor()
        test_images = create_test_images()
        
        results = {}
        
        for evidence_type, image in test_images.items():
            print(f"\nProbando preprocesamiento para {evidence_type}:")
            
            for level in ['basic', 'standard', 'advanced']:
                print(f"  Nivel {level}...")
                start_time = time.time()
                
                try:
                    result = preprocessor.preprocess_image(image, evidence_type, level)
                    processing_time = time.time() - start_time
                    
                    print(f"    ✓ Exitoso - Tiempo: {processing_time:.3f}s")
                    print(f"    ✓ Pasos aplicados: {len(result.steps_applied)}")
                    print(f"    ✓ Calidad: {result.quality_metrics.get('overall_quality', 'N/A')}")
                    
                    results[f"{evidence_type}_{level}"] = {
                        'success': True,
                        'time': processing_time,
                        'steps': len(result.steps_applied),
                        'quality': result.quality_metrics
                    }
                    
                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    results[f"{evidence_type}_{level}"] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
        
    except ImportError as e:
        print(f"✗ Error importando UnifiedPreprocessor: {e}")
        return {}

def test_feature_extraction():
    """Probar la extracción de características"""
    print("\n=== PROBANDO EXTRACCIÓN DE CARACTERÍSTICAS ===")
    
    try:
        from image_processing.ballistic_features_optimized import BallisticFeatureExtractor
        
        extractor = BallisticFeatureExtractor()
        test_images = create_test_images()
        
        results = {}
        
        for evidence_type, image in test_images.items():
            print(f"\nProbando extracción para {evidence_type}:")
            
            start_time = time.time()
            
            try:
                features = extractor.extract_all_features(image)
                processing_time = time.time() - start_time
                
                print(f"  ✓ Exitoso - Tiempo: {processing_time:.3f}s")
                print(f"  ✓ Características extraídas: {len(features)}")
                
                for feature_type, feature_data in features.items():
                    if isinstance(feature_data, dict) and 'roi_count' in feature_data:
                        print(f"    - {feature_type}: {feature_data['roi_count']} ROIs")
                    elif isinstance(feature_data, dict) and 'features' in feature_data:
                        print(f"    - {feature_type}: {len(feature_data['features'])} características")
                    else:
                        print(f"    - {feature_type}: disponible")
                
                results[evidence_type] = {
                    'success': True,
                    'time': processing_time,
                    'features': len(features),
                    'details': features
                }
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results[evidence_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
        
    except ImportError as e:
        print(f"✗ Error importando BallisticFeatureExtractor: {e}")
        return {}

def test_complete_workflow():
    """Probar el flujo completo: preprocesamiento + extracción"""
    print("\n=== PROBANDO FLUJO COMPLETO ===")
    
    try:
        from image_processing.unified_preprocessor import UnifiedPreprocessor
        from image_processing.ballistic_features_optimized import BallisticFeatureExtractor
        
        preprocessor = UnifiedPreprocessor()
        extractor = BallisticFeatureExtractor()
        test_images = create_test_images()
        
        results = {}
        
        for evidence_type, original_image in test_images.items():
            print(f"\nFlujo completo para {evidence_type}:")
            
            total_start = time.time()
            
            try:
                # Paso 1: Preprocesamiento
                print("  1. Preprocesando...")
                preprocess_start = time.time()
                preprocessed_result = preprocessor.preprocess_image(
                    original_image, evidence_type, 'standard'
                )
                preprocess_time = time.time() - preprocess_start
                print(f"     ✓ Preprocesamiento: {preprocess_time:.3f}s")
                
                # Paso 2: Extracción de características
                print("  2. Extrayendo características...")
                extraction_start = time.time()
                features = extractor.extract_all_features(preprocessed_result.processed_image)
                extraction_time = time.time() - extraction_start
                print(f"     ✓ Extracción: {extraction_time:.3f}s")
                
                total_time = time.time() - total_start
                
                print(f"  ✓ Flujo completo exitoso - Tiempo total: {total_time:.3f}s")
                print(f"    - Pasos de preprocesamiento: {len(preprocessed_result.steps_applied)}")
                print(f"    - Características extraídas: {len(features)}")
                
                results[evidence_type] = {
                    'success': True,
                    'total_time': total_time,
                    'preprocess_time': preprocess_time,
                    'extraction_time': extraction_time,
                    'preprocess_steps': len(preprocessed_result.steps_applied),
                    'feature_count': len(features),
                    'quality_metrics': preprocessed_result.quality_metrics
                }
                
            except Exception as e:
                print(f"  ✗ Error en flujo completo: {str(e)}")
                results[evidence_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
        
    except ImportError as e:
        print(f"✗ Error importando módulos: {e}")
        return {}

def generate_integration_report(preprocess_results, extraction_results, workflow_results):
    """Generar reporte de integración"""
    print("\n" + "="*60)
    print("REPORTE DE INTEGRACIÓN COMPLETA")
    print("="*60)
    
    # Resumen de preprocesamiento
    print("\n1. PREPROCESAMIENTO:")
    preprocess_success = sum(1 for r in preprocess_results.values() if r.get('success', False))
    preprocess_total = len(preprocess_results)
    print(f"   Éxito: {preprocess_success}/{preprocess_total} pruebas")
    
    if preprocess_success > 0:
        avg_time = np.mean([r['time'] for r in preprocess_results.values() if r.get('success')])
        print(f"   Tiempo promedio: {avg_time:.3f}s")
    
    # Resumen de extracción
    print("\n2. EXTRACCIÓN DE CARACTERÍSTICAS:")
    extraction_success = sum(1 for r in extraction_results.values() if r.get('success', False))
    extraction_total = len(extraction_results)
    print(f"   Éxito: {extraction_success}/{extraction_total} pruebas")
    
    if extraction_success > 0:
        avg_time = np.mean([r['time'] for r in extraction_results.values() if r.get('success')])
        print(f"   Tiempo promedio: {avg_time:.3f}s")
    
    # Resumen de flujo completo
    print("\n3. FLUJO COMPLETO:")
    workflow_success = sum(1 for r in workflow_results.values() if r.get('success', False))
    workflow_total = len(workflow_results)
    print(f"   Éxito: {workflow_success}/{workflow_total} pruebas")
    
    if workflow_success > 0:
        avg_total_time = np.mean([r['total_time'] for r in workflow_results.values() if r.get('success')])
        avg_preprocess_time = np.mean([r['preprocess_time'] for r in workflow_results.values() if r.get('success')])
        avg_extraction_time = np.mean([r['extraction_time'] for r in workflow_results.values() if r.get('success')])
        
        print(f"   Tiempo total promedio: {avg_total_time:.3f}s")
        print(f"   - Preprocesamiento: {avg_preprocess_time:.3f}s ({avg_preprocess_time/avg_total_time*100:.1f}%)")
        print(f"   - Extracción: {avg_extraction_time:.3f}s ({avg_extraction_time/avg_total_time*100:.1f}%)")
    
    # Estado general
    print(f"\n4. ESTADO GENERAL DEL SISTEMA:")
    total_tests = preprocess_total + extraction_total + workflow_total
    total_success = preprocess_success + extraction_success + workflow_success
    success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    
    print(f"   Tasa de éxito general: {success_rate:.1f}% ({total_success}/{total_tests})")
    
    if success_rate >= 90:
        print("   ✓ SISTEMA FUNCIONANDO CORRECTAMENTE")
    elif success_rate >= 70:
        print("   ⚠ SISTEMA FUNCIONANDO CON PROBLEMAS MENORES")
    else:
        print("   ✗ SISTEMA CON PROBLEMAS SIGNIFICATIVOS")
    
    # Errores encontrados
    errors = []
    for results in [preprocess_results, extraction_results, workflow_results]:
        for test_name, result in results.items():
            if not result.get('success', True):
                errors.append(f"{test_name}: {result.get('error', 'Error desconocido')}")
    
    if errors:
        print(f"\n5. ERRORES ENCONTRADOS ({len(errors)}):")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\n5. ✓ NO SE ENCONTRARON ERRORES")

def main():
    """Función principal"""
    print("Iniciando pruebas de integración completa del sistema SIGeC-Balistica...")
    print("="*60)
    
    # Ejecutar todas las pruebas
    preprocess_results = test_preprocessing()
    extraction_results = test_feature_extraction()
    workflow_results = test_complete_workflow()
    
    # Generar reporte
    generate_integration_report(preprocess_results, extraction_results, workflow_results)
    
    print(f"\n{'='*60}")
    print("PRUEBAS DE INTEGRACIÓN COMPLETADAS")
    print("="*60)

if __name__ == "__main__":
    main()