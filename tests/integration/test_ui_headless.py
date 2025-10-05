#!/usr/bin/env python3
"""
Script de Pruebas Headless para UI - Sistema Balístico Forense MVP
Pruebas sin interfaz gráfica para verificar la funcionalidad del backend de la UI
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar para modo headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from config.unified_config import get_unified_config
from database.vector_db import VectorDatabase
from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingLevel
from image_processing.feature_extractor import FeatureExtractor
from matching.unified_matcher import UnifiedMatcher, MatchingConfig


class HeadlessUITester:
    """Tester para verificar funcionalidad de UI sin interfaz gráfica"""
    
    def __init__(self):
        self.config = get_unified_config()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }
        
    def run_all_tests(self):
        """Ejecutar todas las pruebas headless"""
        print("Iniciando pruebas headless de UI...")
        
        try:
            # Pruebas de componentes backend
            self._test_config_loading()
            self._test_database_initialization()
            self._test_image_processing_pipeline()
            self._test_feature_extraction()
            self._test_matching_functionality()
            self._test_image_loading_simulation()
            
            # Calcular resumen
            self.results['summary']['total'] = len(self.results['tests'])
            self.results['summary']['passed'] = sum(1 for test in self.results['tests'].values() if test['status'] == 'PASSED')
            self.results['summary']['failed'] = self.results['summary']['total'] - self.results['summary']['passed']
            
            self._print_results()
            self._save_results()
            
        except Exception as e:
            self.results['summary']['errors'].append(f"Error general en pruebas: {str(e)}")
            print(f"Error general: {e}")
    
    def _test_config_loading(self):
        """Prueba de carga de configuración"""
        test_name = "config_loading"
        print(f"[20%] Probando carga de configuración...")
        
        try:
            config = get_unified_config()
            
            # Verificar que la configuración se cargó
            if not hasattr(config, 'database') or not config.database:
                raise Exception("Configuración de base de datos no cargada")
            
            if not hasattr(config, 'image_processing') or not config.image_processing:
                raise Exception("Configuración de procesamiento no cargada")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Configuración cargada correctamente',
                'details': {
                    'database_config': bool(config.database),
                    'image_processing_config': bool(config.image_processing),
                    'config_file': config.config_file
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en carga de configuración: {str(e)}',
                'details': {}
            }
    
    def _test_database_initialization(self):
        """Prueba de inicialización de base de datos"""
        test_name = "database_initialization"
        print(f"[40%] Probando inicialización de base de datos...")
        
        try:
            database = VectorDatabase(self.config)
            
            # Verificar que la base de datos se inicializó
            if not database:
                raise Exception("Base de datos no inicializada")
            
            # Probar obtener estadísticas
            stats = database.get_database_stats()
            
            if not isinstance(stats, dict):
                raise Exception("Estadísticas no válidas")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Base de datos inicializada correctamente',
                'details': {
                    'database_path': database.db_path,
                    'statistics': stats
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en inicialización de base de datos: {str(e)}',
                'details': {}
            }
    
    def _test_image_processing_pipeline(self):
        """Prueba del pipeline de procesamiento de imágenes"""
        test_name = "image_processing_pipeline"
        print(f"[60%] Probando pipeline de procesamiento...")
        
        try:
            # Verificar que existe una imagen de prueba
            test_image_path = "uploads/data/images/dekinder/breech_face/SS007_CCI BF R.png"
            if not os.path.exists(test_image_path):
                raise Exception(f"No se encontró imagen de prueba: {test_image_path}")
            
            # Inicializar preprocessor
            preprocessor = UnifiedPreprocessor()
            
            # Procesar imagen
            result = preprocessor.preprocess_ballistic_image(
                test_image_path, 
                evidence_type="vaina",
                level=PreprocessingLevel.STANDARD.value
            )
            
            if not result.success:
                raise Exception(f"Error en preprocesamiento: {result.error_message}")
            
            if result.processed_image is None:
                raise Exception("Imagen procesada es None")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Pipeline de procesamiento funcionando correctamente',
                'details': {
                    'test_image': test_image_path,
                    'processing_success': result.success,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en pipeline de procesamiento: {str(e)}',
                'details': {}
            }
    
    def _test_feature_extraction(self):
        """Prueba de extracción de características"""
        test_name = "feature_extraction"
        print(f"[80%] Probando extracción de características...")
        
        try:
            # Verificar imagen de prueba
            test_image_path = "uploads/data/images/dekinder/breech_face/SS007_CCI BF R.png"
            if not os.path.exists(test_image_path):
                raise Exception(f"No se encontró imagen de prueba: {test_image_path}")
            
            # Procesar imagen primero
            preprocessor = UnifiedPreprocessor()
            preprocess_result = preprocessor.preprocess_ballistic_image(
                test_image_path, 
                evidence_type="vaina",
                level=PreprocessingLevel.STANDARD.value
            )
            
            if not preprocess_result.success:
                raise Exception("Error en preprocesamiento para extracción")
            
            # Extraer características
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_orb_features(preprocess_result.processed_image)
            
            if not features or 'keypoints' not in features:
                raise Exception("No se extrajeron características válidas")
            
            if len(features['keypoints']) == 0:
                raise Exception("No se encontraron keypoints")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Extracción de características funcionando correctamente',
                'details': {
                    'keypoints_count': len(features['keypoints']),
                    'descriptors_shape': features['descriptors'].shape if features['descriptors'] is not None else None,
                    'algorithm': 'ORB'
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en extracción de características: {str(e)}',
                'details': {}
            }
    
    def _test_matching_functionality(self):
        """Prueba de funcionalidad de matching"""
        test_name = "matching_functionality"
        print(f"[90%] Probando funcionalidad de matching...")
        
        try:
            # Inicializar matcher
            matching_config = Matchingget_unified_config()
            matcher = UnifiedMatcher(matching_config)
            
            if not matcher:
                raise Exception("Matcher no inicializado")
            
            # Verificar que tiene los métodos necesarios
            required_methods = ['match_features', 'compare_images']
            for method in required_methods:
                if not hasattr(matcher, method):
                    raise Exception(f"Método {method} no encontrado en matcher")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Funcionalidad de matching verificada correctamente',
                'details': {
                    'matcher_initialized': bool(matcher),
                    'required_methods': required_methods,
                    'config': {
                        'algorithm': matching_config.algorithm.value,
                        'level': matching_config.level.value
                    }
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en funcionalidad de matching: {str(e)}',
                'details': {}
            }
    
    def _test_image_loading_simulation(self):
        """Simulación de carga de imagen como lo haría la UI"""
        test_name = "image_loading_simulation"
        print(f"[100%] Simulando carga de imagen de la UI...")
        
        try:
            # Simular el flujo completo de la UI
            test_image_path = "uploads/data/images/dekinder/breech_face/SS007_CCI BF R.png"
            
            # 1. Verificar que el archivo existe (como haría la UI)
            if not os.path.exists(test_image_path):
                raise Exception("Archivo de imagen no existe")
            
            # 2. Verificar que es un formato válido
            valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
            file_ext = Path(test_image_path).suffix.lower()
            if file_ext not in valid_extensions:
                raise Exception(f"Formato de archivo no válido: {file_ext}")
            
            # 3. Obtener información del archivo
            file_size = os.path.getsize(test_image_path)
            if file_size == 0:
                raise Exception("Archivo de imagen vacío")
            
            # 4. Simular procesamiento completo
            preprocessor = UnifiedPreprocessor()
            result = preprocessor.preprocess_ballistic_image(test_image_path, evidence_type="vaina")
            
            if not result.success:
                raise Exception("Error en procesamiento simulado")
            
            self.results['tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Simulación de carga de imagen exitosa',
                'details': {
                    'file_path': test_image_path,
                    'file_size': file_size,
                    'file_extension': file_ext,
                    'processing_success': result.success,
                    'processing_time': result.processing_time
                }
            }
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en simulación de carga: {str(e)}',
                'details': {}
            }
    
    def _print_results(self):
        """Imprimir resultados de las pruebas"""
        print("\n" + "="*60)
        print("RESULTADOS DE PRUEBAS HEADLESS DE UI")
        print("="*60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Total de pruebas: {self.results['summary']['total']}")
        print(f"Pruebas exitosas: {self.results['summary']['passed']}")
        print(f"Pruebas fallidas: {self.results['summary']['failed']}")
        
        if self.results['summary']['errors']:
            print(f"Errores generales: {len(self.results['summary']['errors'])}")
            for error in self.results['summary']['errors']:
                print(f"  - {error}")
        
        print("\nDetalle de pruebas:")
        print("-" * 40)
        
        for test_name, test_result in self.results['tests'].items():
            status_symbol = "✅" if test_result['status'] == 'PASSED' else "❌"
            print(f"{status_symbol} {test_name}: {test_result['status']}")
            print(f"   Mensaje: {test_result['message']}")
            if test_result['details']:
                print(f"   Detalles: {json.dumps(test_result['details'], indent=6)}")
            print()
    
    def _save_results(self):
        """Guardar resultados en archivo"""
        results_file = f"headless_ui_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en: {results_file}")


def main():
    """Función principal"""
    try:
        tester = HeadlessUITester()
        tester.run_all_tests()
        
        # Determinar código de salida basado en resultados
        if tester.results['summary']['failed'] > 0:
            print(f"\n❌ {tester.results['summary']['failed']} pruebas fallaron")
            sys.exit(1)
        else:
            print(f"\n✅ Todas las {tester.results['summary']['passed']} pruebas pasaron")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error al ejecutar pruebas headless: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()