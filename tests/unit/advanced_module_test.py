#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Pruebas Avanzadas para Módulos de Procesamiento de Imágenes
Sistema Balístico Forense SIGeC-Balistica

Este script prueba funcionalidades avanzadas de cada módulo individualmente
para identificar problemas específicos y oportunidades de mejora.
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append('/home/marco/SIGeC-Balistica')

class AdvancedModuleTester:
    """Clase para pruebas avanzadas de módulos individuales"""
    
    def __init__(self, samples_dir: str = "uploads/Muestras NIST FADB"):
        self.samples_dir = Path(samples_dir)
        self.results_dir = Path("advanced_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Crear subdirectorios para resultados
        (self.results_dir / "module_tests").mkdir(exist_ok=True)
        (self.results_dir / "performance").mkdir(exist_ok=True)
        (self.results_dir / "issues").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'module_tests': {},
            'performance_metrics': {},
            'identified_issues': [],
            'improvement_suggestions': [],
            'test_images': []
        }
        
        print(f"Advanced Tester inicializado. Directorio de muestras: {self.samples_dir}")
        print(f"Directorio de resultados: {self.results_dir}")
    
    def get_test_images(self, max_images: int = 5) -> List[Path]:
        """Obtiene imágenes de prueba de diferentes tipos"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        test_images = []
        
        try:
            # Buscar imágenes con diferentes características
            for root, dirs, files in os.walk(self.samples_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_path = Path(root) / file
                        
                        # Categorizar por tipo de imagen basado en el nombre
                        image_type = "unknown"
                        if "BF" in file:  # Breech Face
                            image_type = "breech_face"
                        elif "FP" in file:  # Firing Pin
                            image_type = "firing_pin"
                        elif "3DVM" in file:  # 3D Virtual Microscopy
                            image_type = "3d_microscopy"
                        
                        test_images.append({
                            'path': image_path,
                            'type': image_type,
                            'name': file
                        })
                        
                        if len(test_images) >= max_images:
                            break
                
                if len(test_images) >= max_images:
                    break
            
            print(f"Encontradas {len(test_images)} imágenes de prueba")
            return test_images[:max_images]
            
        except Exception as e:
            print(f"Error buscando imágenes de prueba: {e}")
            return []
    
    def test_feature_extractor_module(self, test_images: List[Dict]) -> Dict[str, Any]:
        """Prueba el módulo feature_extractor.py"""
        print("\n=== Probando feature_extractor.py ===")
        
        module_result = {
            'module_name': 'feature_extractor',
            'import_success': False,
            'class_tests': {},
            'function_tests': {},
            'performance': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Intentar importar el módulo
            from image_processing.feature_extractor import FeatureExtractor, extract_features
            module_result['import_success'] = True
            print("✓ Importación exitosa")
            
            # Probar la clase FeatureExtractor
            try:
                extractor = FeatureExtractor()
                module_result['class_tests']['FeatureExtractor'] = {
                    'instantiation': True,
                    'methods_tested': {}
                }
                print("✓ Instanciación de FeatureExtractor exitosa")
                
                # Probar métodos con imágenes reales
                for img_data in test_images[:2]:  # Probar con 2 imágenes
                    img_path = img_data['path']
                    img_name = img_data['name']
                    
                    print(f"  Probando con: {img_name}")
                    
                    # Cargar imagen
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Probar extracción SIFT
                    try:
                        start_time = time.time()
                        sift_result = extractor.extract_sift_features(gray)
                        sift_time = time.time() - start_time
                        
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_sift_features'] = {
                            'success': True,
                            'processing_time': sift_time,
                            'keypoints_found': len(sift_result.get('sift_keypoints', [])),
                            'descriptors_shape': sift_result.get('sift_descriptors', np.array([])).shape if 'sift_descriptors' in sift_result else None
                        }
                        print(f"    ✓ SIFT: {len(sift_result.get('sift_keypoints', []))} keypoints en {sift_time:.3f}s")
                        
                    except Exception as e:
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_sift_features'] = {
                            'success': False,
                            'error': str(e)
                        }
                        module_result['issues'].append(f"Error en SIFT: {str(e)}")
                        print(f"    ✗ Error en SIFT: {e}")
                    
                    # Probar extracción ORB
                    try:
                        start_time = time.time()
                        orb_result = extractor.extract_orb_features(gray)
                        orb_time = time.time() - start_time
                        
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_orb_features'] = {
                            'success': True,
                            'processing_time': orb_time,
                            'keypoints_found': len(orb_result.get('orb_keypoints', [])),
                            'descriptors_shape': orb_result.get('orb_descriptors', np.array([])).shape if 'orb_descriptors' in orb_result else None
                        }
                        print(f"    ✓ ORB: {len(orb_result.get('orb_keypoints', []))} keypoints en {orb_time:.3f}s")
                        
                    except Exception as e:
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_orb_features'] = {
                            'success': False,
                            'error': str(e)
                        }
                        module_result['issues'].append(f"Error en ORB: {str(e)}")
                        print(f"    ✗ Error en ORB: {e}")
                    
                    # Probar extracción LBP
                    try:
                        start_time = time.time()
                        lbp_result = extractor.extract_lbp_features(gray)
                        lbp_time = time.time() - start_time
                        
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_lbp_features'] = {
                            'success': True,
                            'processing_time': lbp_time,
                            'histogram_size': len(lbp_result.get('lbp_histogram', [])),
                            'uniformity': lbp_result.get('lbp_uniformity', 0)
                        }
                        print(f"    ✓ LBP: histograma de {len(lbp_result.get('lbp_histogram', []))} bins en {lbp_time:.3f}s")
                        
                    except Exception as e:
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_lbp_features'] = {
                            'success': False,
                            'error': str(e)
                        }
                        module_result['issues'].append(f"Error en LBP: {str(e)}")
                        print(f"    ✗ Error en LBP: {e}")
                    
                    # Probar extracción Gabor
                    try:
                        start_time = time.time()
                        gabor_result = extractor.extract_gabor_features(gray)
                        gabor_time = time.time() - start_time
                        
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_gabor_features'] = {
                            'success': True,
                            'processing_time': gabor_time,
                            'responses_count': len(gabor_result.get('gabor_responses', [])),
                            'mean_response': np.mean(gabor_result.get('gabor_responses', [])) if gabor_result.get('gabor_responses') else 0
                        }
                        print(f"    ✓ Gabor: {len(gabor_result.get('gabor_responses', []))} respuestas en {gabor_time:.3f}s")
                        
                    except Exception as e:
                        module_result['class_tests']['FeatureExtractor']['methods_tested']['extract_gabor_features'] = {
                            'success': False,
                            'error': str(e)
                        }
                        module_result['issues'].append(f"Error en Gabor: {str(e)}")
                        print(f"    ✗ Error en Gabor: {e}")
                    
                    break  # Solo probar con la primera imagen válida
                        
            except Exception as e:
                module_result['class_tests']['FeatureExtractor'] = {
                    'instantiation': False,
                    'error': str(e)
                }
                module_result['issues'].append(f"Error instanciando FeatureExtractor: {str(e)}")
                print(f"✗ Error instanciando FeatureExtractor: {e}")
            
            # Probar función extract_features
            try:
                if test_images:
                    img_path = str(test_images[0]['path'])
                    start_time = time.time()
                    features = extract_features(img_path)
                    extract_time = time.time() - start_time
                    
                    module_result['function_tests']['extract_features'] = {
                        'success': True,
                        'processing_time': extract_time,
                        'features_returned': len(features) if isinstance(features, dict) else 0
                    }
                    print(f"✓ extract_features: procesado en {extract_time:.3f}s")
                    
            except Exception as e:
                module_result['function_tests']['extract_features'] = {
                    'success': False,
                    'error': str(e)
                }
                module_result['issues'].append(f"Error en extract_features: {str(e)}")
                print(f"✗ Error en extract_features: {e}")
        
        except ImportError as e:
            module_result['issues'].append(f"Error de importación: {str(e)}")
            print(f"✗ Error de importación: {e}")
        except Exception as e:
            module_result['issues'].append(f"Error general: {str(e)}")
            print(f"✗ Error general: {e}")
        
        # Generar recomendaciones específicas
        if module_result['import_success']:
            if len(module_result['issues']) == 0:
                module_result['recommendations'].append("Módulo funcionando correctamente")
            else:
                module_result['recommendations'].append("Revisar errores específicos en métodos individuales")
                
            # Analizar rendimiento
            method_times = []
            for method_name, method_data in module_result['class_tests'].get('FeatureExtractor', {}).get('methods_tested', {}).items():
                if method_data.get('success') and 'processing_time' in method_data:
                    method_times.append(method_data['processing_time'])
            
            if method_times:
                avg_time = np.mean(method_times)
                if avg_time > 2.0:
                    module_result['recommendations'].append("Considerar optimización de rendimiento (tiempo promedio > 2s)")
                elif avg_time < 0.5:
                    module_result['recommendations'].append("Rendimiento excelente (tiempo promedio < 0.5s)")
        
        return module_result
    
    def test_ballistic_features_module(self, test_images: List[Dict]) -> Dict[str, Any]:
        """Prueba el módulo ballistic_features.py"""
        print("\n=== Probando ballistic_features.py ===")
        
        module_result = {
            'module_name': 'ballistic_features',
            'import_success': False,
            'class_tests': {},
            'function_tests': {},
            'performance': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Intentar importar el módulo
            from image_processing.ballistic_features import BallisticFeatureExtractor, extract_ballistic_features_from_path
            module_result['import_success'] = True
            print("✓ Importación exitosa")
            
            # Probar la clase BallisticFeatureExtractor
            try:
                extractor = BallisticFeatureExtractor()
                module_result['class_tests']['BallisticFeatureExtractor'] = {
                    'instantiation': True,
                    'methods_tested': {}
                }
                print("✓ Instanciación de BallisticFeatureExtractor exitosa")
                
                # Probar con imágenes reales
                for img_data in test_images[:2]:
                    img_path = img_data['path']
                    img_name = img_data['name']
                    img_type = img_data['type']
                    
                    print(f"  Probando con: {img_name} (tipo: {img_type})")
                    
                    # Cargar imagen
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    # Probar extracción de características específicas según el tipo
                    if img_type == "breech_face":
                        try:
                            start_time = time.time()
                            bf_features = extractor.extract_breech_face_features(image)
                            bf_time = time.time() - start_time
                            
                            module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_breech_face_features'] = {
                                'success': True,
                                'processing_time': bf_time,
                                'features_extracted': len(bf_features) if isinstance(bf_features, dict) else 0
                            }
                            print(f"    ✓ Breech Face features: procesado en {bf_time:.3f}s")
                            
                        except Exception as e:
                            module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_breech_face_features'] = {
                                'success': False,
                                'error': str(e)
                            }
                            module_result['issues'].append(f"Error en breech face features: {str(e)}")
                            print(f"    ✗ Error en breech face features: {e}")
                    
                    elif img_type == "firing_pin":
                        try:
                            start_time = time.time()
                            fp_features = extractor.extract_firing_pin_features(image)
                            fp_time = time.time() - start_time
                            
                            module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_firing_pin_features'] = {
                                'success': True,
                                'processing_time': fp_time,
                                'features_extracted': len(fp_features) if isinstance(fp_features, dict) else 0
                            }
                            print(f"    ✓ Firing Pin features: procesado en {fp_time:.3f}s")
                            
                        except Exception as e:
                            module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_firing_pin_features'] = {
                                'success': False,
                                'error': str(e)
                            }
                            module_result['issues'].append(f"Error en firing pin features: {str(e)}")
                            print(f"    ✗ Error en firing pin features: {e}")
                    
                    # Probar extracción general de características balísticas
                    try:
                        start_time = time.time()
                        general_features = extractor.extract_all_features(image)
                        general_time = time.time() - start_time
                        
                        module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_all_features'] = {
                            'success': True,
                            'processing_time': general_time,
                            'features_extracted': len(general_features) if isinstance(general_features, dict) else 0
                        }
                        print(f"    ✓ All features: procesado en {general_time:.3f}s")
                        
                    except Exception as e:
                        module_result['class_tests']['BallisticFeatureExtractor']['methods_tested']['extract_all_features'] = {
                            'success': False,
                            'error': str(e)
                        }
                        module_result['issues'].append(f"Error en extract_all_features: {str(e)}")
                        print(f"    ✗ Error en extract_all_features: {e}")
                    
                    break  # Solo probar con la primera imagen válida
                        
            except Exception as e:
                module_result['class_tests']['BallisticFeatureExtractor'] = {
                    'instantiation': False,
                    'error': str(e)
                }
                module_result['issues'].append(f"Error instanciando BallisticFeatureExtractor: {str(e)}")
                print(f"✗ Error instanciando BallisticFeatureExtractor: {e}")
            
            # Probar función extract_ballistic_features_from_path
            try:
                if test_images:
                    img_path = str(test_images[0]['path'])
                    start_time = time.time()
                    features = extract_ballistic_features_from_path(img_path)
                    extract_time = time.time() - start_time
                    
                    module_result['function_tests']['extract_ballistic_features_from_path'] = {
                        'success': True,
                        'processing_time': extract_time,
                        'features_returned': len(features) if isinstance(features, dict) else 0,
                        'success_flag': features.get('success', False) if isinstance(features, dict) else False
                    }
                    print(f"✓ extract_ballistic_features_from_path: procesado en {extract_time:.3f}s")
                    
            except Exception as e:
                module_result['function_tests']['extract_ballistic_features_from_path'] = {
                    'success': False,
                    'error': str(e)
                }
                module_result['issues'].append(f"Error en extract_ballistic_features_from_path: {str(e)}")
                print(f"✗ Error en extract_ballistic_features_from_path: {e}")
        
        except ImportError as e:
            module_result['issues'].append(f"Error de importación: {str(e)}")
            print(f"✗ Error de importación: {e}")
        except Exception as e:
            module_result['issues'].append(f"Error general: {str(e)}")
            print(f"✗ Error general: {e}")
        
        # Generar recomendaciones específicas
        if module_result['import_success']:
            if len(module_result['issues']) == 0:
                module_result['recommendations'].append("Módulo funcionando correctamente")
            else:
                module_result['recommendations'].append("Revisar errores específicos en métodos balísticos")
                
            # Verificar si se detectaron características específicas por tipo de imagen
            breech_face_tested = any('breech_face' in method for method in module_result['class_tests'].get('BallisticFeatureExtractor', {}).get('methods_tested', {}))
            firing_pin_tested = any('firing_pin' in method for method in module_result['class_tests'].get('BallisticFeatureExtractor', {}).get('methods_tested', {}))
            
            if not breech_face_tested:
                module_result['recommendations'].append("Probar con imágenes de breech face para validación completa")
            if not firing_pin_tested:
                module_result['recommendations'].append("Probar con imágenes de firing pin para validación completa")
        
        return module_result
    
    def test_unified_preprocessor_module(self, test_images: List[Dict]) -> Dict[str, Any]:
        """Prueba el módulo unified_preprocessor.py"""
        print("\n=== Probando unified_preprocessor.py ===")
        
        module_result = {
            'module_name': 'unified_preprocessor',
            'import_success': False,
            'class_tests': {},
            'performance': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Intentar importar el módulo
            from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingLevel
            module_result['import_success'] = True
            print("✓ Importación exitosa")
            
            # Probar la clase UnifiedPreprocessor
            try:
                preprocessor = UnifiedPreprocessor()
                module_result['class_tests']['UnifiedPreprocessor'] = {
                    'instantiation': True,
                    'methods_tested': {}
                }
                print("✓ Instanciación de UnifiedPreprocessor exitosa")
                
                # Probar con diferentes niveles de preprocesamiento
                if test_images:
                    img_path = str(test_images[0]['path'])
                    
                    for level in ['basic', 'standard', 'advanced']:
                        try:
                            print(f"  Probando nivel: {level}")
                            start_time = time.time()
                            
                            result = preprocessor.preprocess_image(
                                img_path,
                                evidence_type='cartridge_case',
                                level=level
                            )
                            
                            process_time = time.time() - start_time
                            
                            module_result['class_tests']['UnifiedPreprocessor']['methods_tested'][f'preprocess_level_{level}'] = {
                                'success': True,
                                'processing_time': process_time,
                                'result_type': type(result).__name__,
                                'has_processed_image': hasattr(result, 'processed_image') if hasattr(result, '__dict__') else False
                            }
                            print(f"    ✓ Nivel {level}: procesado en {process_time:.3f}s")
                            
                        except Exception as e:
                            module_result['class_tests']['UnifiedPreprocessor']['methods_tested'][f'preprocess_level_{level}'] = {
                                'success': False,
                                'error': str(e)
                            }
                            module_result['issues'].append(f"Error en nivel {level}: {str(e)}")
                            print(f"    ✗ Error en nivel {level}: {e}")
                        
            except Exception as e:
                module_result['class_tests']['UnifiedPreprocessor'] = {
                    'instantiation': False,
                    'error': str(e)
                }
                module_result['issues'].append(f"Error instanciando UnifiedPreprocessor: {str(e)}")
                print(f"✗ Error instanciando UnifiedPreprocessor: {e}")
        
        except ImportError as e:
            module_result['issues'].append(f"Error de importación: {str(e)}")
            print(f"✗ Error de importación: {e}")
        except Exception as e:
            module_result['issues'].append(f"Error general: {str(e)}")
            print(f"✗ Error general: {e}")
        
        # Generar recomendaciones
        if module_result['import_success']:
            if len(module_result['issues']) == 0:
                module_result['recommendations'].append("Módulo de preprocesamiento funcionando correctamente")
            else:
                module_result['recommendations'].append("Revisar configuración de niveles de preprocesamiento")
        
        return module_result
    
    def run_advanced_tests(self) -> Dict[str, Any]:
        """Ejecuta todas las pruebas avanzadas"""
        print("Iniciando pruebas avanzadas de módulos individuales")
        print("=" * 60)
        
        # Obtener imágenes de prueba
        test_images = self.get_test_images(max_images=5)
        self.test_results['test_images'] = [{'name': img['name'], 'type': img['type']} for img in test_images]
        
        if not test_images:
            print("No se encontraron imágenes de prueba")
            return self.test_results
        
        # Probar cada módulo
        modules_to_test = [
            ('feature_extractor', self.test_feature_extractor_module),
            ('ballistic_features', self.test_ballistic_features_module),
            ('unified_preprocessor', self.test_unified_preprocessor_module)
        ]
        
        for module_name, test_function in modules_to_test:
            try:
                print(f"\n{'='*60}")
                result = test_function(test_images)
                self.test_results['module_tests'][module_name] = result
                
                # Agregar issues globales
                if result['issues']:
                    self.test_results['identified_issues'].extend([
                        f"{module_name}: {issue}" for issue in result['issues']
                    ])
                
                # Agregar recomendaciones globales
                if result['recommendations']:
                    self.test_results['improvement_suggestions'].extend([
                        f"{module_name}: {rec}" for rec in result['recommendations']
                    ])
                
            except Exception as e:
                error_msg = f"Error crítico probando {module_name}: {str(e)}"
                print(f"✗ {error_msg}")
                self.test_results['identified_issues'].append(error_msg)
        
        # Guardar resultados
        self.save_advanced_results()
        
        return self.test_results
    
    def save_advanced_results(self):
        """Guarda los resultados de las pruebas avanzadas"""
        # Guardar JSON completo
        results_file = self.results_dir / "advanced_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Crear reporte detallado
        report_file = self.results_dir / "advanced_test_report.txt"
        with open(report_file, 'w') as f:
            f.write("REPORTE DE PRUEBAS AVANZADAS - MÓDULOS DE PROCESAMIENTO\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"Fecha: {self.test_results['timestamp']}\n")
            f.write(f"Imágenes de prueba: {len(self.test_results['test_images'])}\n\n")
            
            # Resumen por módulo
            f.write("RESUMEN POR MÓDULO:\n")
            f.write("-" * 30 + "\n")
            for module_name, module_data in self.test_results['module_tests'].items():
                f.write(f"\n{module_name.upper()}:\n")
                f.write(f"  Importación: {'✓' if module_data['import_success'] else '✗'}\n")
                f.write(f"  Issues encontrados: {len(module_data['issues'])}\n")
                f.write(f"  Recomendaciones: {len(module_data['recommendations'])}\n")
                
                # Detalles de clases probadas
                for class_name, class_data in module_data.get('class_tests', {}).items():
                    if isinstance(class_data, dict) and class_data.get('instantiation'):
                        methods_tested = len(class_data.get('methods_tested', {}))
                        successful_methods = sum(1 for method_data in class_data.get('methods_tested', {}).values() 
                                               if isinstance(method_data, dict) and method_data.get('success'))
                        f.write(f"  {class_name}: {successful_methods}/{methods_tested} métodos exitosos\n")
            
            # Issues identificados
            if self.test_results['identified_issues']:
                f.write(f"\nISSUES IDENTIFICADOS ({len(self.test_results['identified_issues'])}):\n")
                f.write("-" * 30 + "\n")
                for i, issue in enumerate(self.test_results['identified_issues'], 1):
                    f.write(f"{i}. {issue}\n")
            
            # Recomendaciones
            if self.test_results['improvement_suggestions']:
                f.write(f"\nRECOMENDACIONES DE MEJORA ({len(self.test_results['improvement_suggestions'])}):\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(self.test_results['improvement_suggestions'], 1):
                    f.write(f"{i}. {rec}\n")
        
        print(f"\nResultados avanzados guardados en: {results_file}")
        print(f"Reporte detallado guardado en: {report_file}")

def main():
    """Función principal"""
    print("Iniciando pruebas avanzadas de módulos de procesamiento de imágenes...")
    print("=" * 70)
    
    # Crear tester avanzado
    tester = AdvancedModuleTester()
    
    # Ejecutar pruebas
    results = tester.run_advanced_tests()
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE PRUEBAS AVANZADAS:")
    print("-" * 35)
    
    modules_tested = len(results['module_tests'])
    successful_imports = sum(1 for module_data in results['module_tests'].values() 
                           if module_data['import_success'])
    total_issues = len(results['identified_issues'])
    total_recommendations = len(results['improvement_suggestions'])
    
    print(f"Módulos probados: {modules_tested}")
    print(f"Importaciones exitosas: {successful_imports}/{modules_tested}")
    print(f"Issues identificados: {total_issues}")
    print(f"Recomendaciones generadas: {total_recommendations}")
    
    if total_issues > 0:
        print(f"\nPrincipales issues:")
        for i, issue in enumerate(results['identified_issues'][:3], 1):
            print(f"  {i}. {issue}")
    
    if total_recommendations > 0:
        print(f"\nPrincipales recomendaciones:")
        for i, rec in enumerate(results['improvement_suggestions'][:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nResultados detallados en: {tester.results_dir}")

if __name__ == "__main__":
    main()