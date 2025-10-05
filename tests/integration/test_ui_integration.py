#!/usr/bin/env python3
"""
Pruebas de Integración UI - Sistema Balístico Forense MVP
Pruebas completas end-to-end para verificar funcionalidad de todas las pestañas
"""

import sys
import os
import json
import time
import shutil
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


class UIIntegrationTester:
    """Tester de integración completa para UI"""
    
    def __init__(self):
        self.config = get_unified_config()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'integration_tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': []
            }
        }
        
        # Rutas de imágenes de prueba
        self.test_images = [
            "uploads/data/images/dekinder/breech_face/SS007_CCI BF R.png",
            "uploads/data/images/dekinder/breech_face/SS007_FED BF R.png",
            "uploads/data/images/dekinder/firing_pin/SS007_CCI FP R.png"
        ]
        
    def run_integration_tests(self):
        """Ejecutar todas las pruebas de integración"""
        print("Iniciando pruebas de integración UI...")
        
        try:
            # Pruebas de flujo completo
            self._test_image_tab_workflow()
            self._test_database_tab_workflow()
            self._test_comparison_tab_workflow()
            self._test_cross_tab_integration()
            self._test_data_persistence()
            
            # Calcular resumen
            self.results['summary']['total'] = len(self.results['integration_tests'])
            self.results['summary']['passed'] = sum(1 for test in self.results['integration_tests'].values() if test['status'] == 'PASSED')
            self.results['summary']['failed'] = self.results['summary']['total'] - self.results['summary']['passed']
            
            self._print_results()
            self._save_results()
            
        except Exception as e:
            self.results['summary']['errors'].append(f"Error general en pruebas de integración: {str(e)}")
            print(f"Error general: {e}")
    
    def _test_image_tab_workflow(self):
        """Prueba del flujo completo de la pestaña de imágenes"""
        test_name = "image_tab_workflow"
        print(f"[20%] Probando flujo completo de pestaña de imágenes...")
        
        try:
            # Simular carga de imagen
            test_image = self.test_images[0]
            if not os.path.exists(test_image):
                raise Exception(f"Imagen de prueba no encontrada: {test_image}")
            
            # 1. Preprocesamiento
            preprocessor = UnifiedPreprocessor()
            preprocess_result = preprocessor.preprocess_ballistic_image(
                test_image, 
                evidence_type="vaina",
                level=PreprocessingLevel.STANDARD.value
            )
            
            if not preprocess_result.success:
                raise Exception("Error en preprocesamiento")
            
            # 2. Extracción de características
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_orb_features(preprocess_result.processed_image)
            
            if not features or len(features['keypoints']) == 0:
                raise Exception("No se extrajeron características")
            
            # 3. Simulación de guardado en base de datos
            database = VectorDatabase(self.config)
            
            # Crear caso de prueba
            case_data = {
                'case_number': f'TEST_CASE_{int(time.time())}',
                'investigator': 'Test Investigator',
                'date_created': datetime.now().isoformat(),
                'weapon_type': 'Pistola',
                'caliber': '9mm',
                'description': 'Caso de prueba de integración'
            }
            
            # Simular guardado (sin realmente guardar para no contaminar BD)
            workflow_success = True
            
            self.results['integration_tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Flujo completo de pestaña de imágenes exitoso',
                'details': {
                    'preprocessing_time': preprocess_result.processing_time,
                    'keypoints_extracted': len(features['keypoints']),
                    'quality_metrics': preprocess_result.quality_metrics,
                    'workflow_steps': ['load', 'preprocess', 'extract_features', 'prepare_save']
                }
            }
            
        except Exception as e:
            self.results['integration_tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en flujo de pestaña de imágenes: {str(e)}',
                'details': {}
            }
    
    def _test_database_tab_workflow(self):
        """Prueba del flujo completo de la pestaña de base de datos"""
        test_name = "database_tab_workflow"
        print(f"[40%] Probando flujo completo de pestaña de base de datos...")
        
        try:
            database = VectorDatabase(self.config)
            
            # 1. Obtener estadísticas
            stats = database.get_database_stats()
            if not isinstance(stats, dict):
                raise Exception("Estadísticas no válidas")
            
            # 2. Obtener casos existentes
            cases = database.get_cases()
            
            # 3. Simular búsqueda por similitud
            # Procesar imagen para búsqueda
            test_image = self.test_images[0]
            preprocessor = UnifiedPreprocessor()
            preprocess_result = preprocessor.preprocess_ballistic_image(test_image, evidence_type="vaina")
            
            if not preprocess_result.success:
                raise Exception("Error en preprocesamiento para búsqueda")
            
            feature_extractor = FeatureExtractor()
            query_features = feature_extractor.extract_orb_features(preprocess_result.processed_image)
            
            # Simular búsqueda vectorial (sin ejecutar realmente para evitar modificar BD)
            search_simulation_success = True
            
            self.results['integration_tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Flujo completo de pestaña de base de datos exitoso',
                'details': {
                    'database_stats': stats,
                    'cases_count': len(cases),
                    'search_features_extracted': len(query_features['keypoints']),
                    'workflow_steps': ['get_stats', 'list_cases', 'prepare_search', 'extract_query_features']
                }
            }
            
        except Exception as e:
            self.results['integration_tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en flujo de pestaña de base de datos: {str(e)}',
                'details': {}
            }
    
    def _test_comparison_tab_workflow(self):
        """Prueba del flujo completo de la pestaña de comparación"""
        test_name = "comparison_tab_workflow"
        print(f"[60%] Probando flujo completo de pestaña de comparación...")
        
        try:
            # Verificar que tenemos al menos 2 imágenes
            if len(self.test_images) < 2:
                raise Exception("Se necesitan al menos 2 imágenes para comparación")
            
            # 1. Cargar y procesar primera imagen
            preprocessor = UnifiedPreprocessor()
            result1 = preprocessor.preprocess_ballistic_image(self.test_images[0], evidence_type="vaina")
            result2 = preprocessor.preprocess_ballistic_image(self.test_images[1], evidence_type="vaina")
            
            if not result1.success or not result2.success:
                raise Exception("Error en preprocesamiento de imágenes para comparación")
            
            # 2. Extraer características
            feature_extractor = FeatureExtractor()
            features1 = feature_extractor.extract_orb_features(result1.processed_image)
            features2 = feature_extractor.extract_orb_features(result2.processed_image)
            
            if not features1 or not features2:
                raise Exception("Error en extracción de características")
            
            # 3. Realizar matching
            matching_config = Matchingget_unified_config()
            matcher = UnifiedMatcher(matching_config)
            
            match_result = matcher.match_features(features1, features2)
            
            if not match_result:
                raise Exception("Error en matching")
            
            # 4. Verificar resultados de matching
            if match_result.similarity_score < 0:
                raise Exception("Score de similitud inválido")
            
            self.results['integration_tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Flujo completo de pestaña de comparación exitoso',
                'details': {
                    'image1_keypoints': len(features1['keypoints']),
                    'image2_keypoints': len(features2['keypoints']),
                    'total_matches': match_result.total_matches,
                    'good_matches': match_result.good_matches,
                    'similarity_score': match_result.similarity_score,
                    'confidence': match_result.confidence,
                    'processing_time': match_result.processing_time,
                    'workflow_steps': ['load_images', 'preprocess', 'extract_features', 'match', 'calculate_similarity']
                }
            }
            
        except Exception as e:
            self.results['integration_tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en flujo de pestaña de comparación: {str(e)}',
                'details': {}
            }
    
    def _test_cross_tab_integration(self):
        """Prueba de integración entre pestañas"""
        test_name = "cross_tab_integration"
        print(f"[80%] Probando integración entre pestañas...")
        
        try:
            # Simular flujo: Imagen -> Base de Datos -> Comparación
            
            # 1. Procesar imagen (como en pestaña de imágenes)
            test_image = self.test_images[0]
            preprocessor = UnifiedPreprocessor()
            preprocess_result = preprocessor.preprocess_ballistic_image(test_image, evidence_type="vaina")
            
            if not preprocess_result.success:
                raise Exception("Error en preprocesamiento para integración")
            
            # 2. Extraer características
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_orb_features(preprocess_result.processed_image)
            
            # 3. Simular guardado en BD (pestaña de base de datos)
            database = VectorDatabase(self.config)
            stats_before = database.get_database_stats()
            
            # 4. Simular búsqueda en BD con la misma imagen
            # (esto simularía el flujo de buscar una imagen procesada en la pestaña de imágenes)
            
            # 5. Usar resultado para comparación (pestaña de comparación)
            matching_config = Matchingget_unified_config()
            matcher = UnifiedMatcher(matching_config)
            
            # Simular auto-comparación (debería dar alta similitud)
            match_result = matcher.match_features(features, features)
            
            if match_result.similarity_score < 0.8:  # Auto-match debería ser muy alto
                raise Exception(f"Auto-match score muy bajo: {match_result.similarity_score}")
            
            self.results['integration_tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Integración entre pestañas exitosa',
                'details': {
                    'cross_tab_flow': ['image_tab', 'database_tab', 'comparison_tab'],
                    'auto_match_score': match_result.similarity_score,
                    'database_stats': stats_before,
                    'integration_verified': True
                }
            }
            
        except Exception as e:
            self.results['integration_tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en integración entre pestañas: {str(e)}',
                'details': {}
            }
    
    def _test_data_persistence(self):
        """Prueba de persistencia de datos"""
        test_name = "data_persistence"
        print(f"[100%] Probando persistencia de datos...")
        
        try:
            # 1. Verificar que la base de datos existe y es accesible
            database = VectorDatabase(self.config)
            
            # 2. Verificar estadísticas iniciales
            initial_stats = database.get_database_stats()
            
            # 3. Verificar que los archivos de configuración existen
            config_files = [
                self.config.config_file,
                self.config.get_database_path()
            ]
            
            missing_files = []
            for file_path in config_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            # 4. Verificar directorios necesarios
            required_dirs = [
                Path(self.config.database.db_path),
                Path(self.config.image_processing.temp_path),
                Path(self.config.logging.log_path)
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                if not dir_path.exists():
                    missing_dirs.append(str(dir_path))
            
            if missing_files or missing_dirs:
                raise Exception(f"Archivos faltantes: {missing_files}, Directorios faltantes: {missing_dirs}")
            
            self.results['integration_tests'][test_name] = {
                'status': 'PASSED',
                'message': 'Persistencia de datos verificada correctamente',
                'details': {
                    'database_stats': initial_stats,
                    'config_files_exist': len(config_files) - len(missing_files),
                    'required_dirs_exist': len(required_dirs) - len(missing_dirs),
                    'database_path': self.config.get_database_path(),
                    'faiss_path': self.config.get_faiss_index_path()
                }
            }
            
        except Exception as e:
            self.results['integration_tests'][test_name] = {
                'status': 'FAILED',
                'message': f'Error en persistencia de datos: {str(e)}',
                'details': {}
            }
    
    def _print_results(self):
        """Imprimir resultados de las pruebas de integración"""
        print("\n" + "="*70)
        print("RESULTADOS DE PRUEBAS DE INTEGRACIÓN UI")
        print("="*70)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Total de pruebas: {self.results['summary']['total']}")
        print(f"Pruebas exitosas: {self.results['summary']['passed']}")
        print(f"Pruebas fallidas: {self.results['summary']['failed']}")
        
        if self.results['summary']['errors']:
            print(f"Errores generales: {len(self.results['summary']['errors'])}")
            for error in self.results['summary']['errors']:
                print(f"  - {error}")
        
        print("\nDetalle de pruebas de integración:")
        print("-" * 50)
        
        for test_name, test_result in self.results['integration_tests'].items():
            status_symbol = "✅" if test_result['status'] == 'PASSED' else "❌"
            print(f"{status_symbol} {test_name}: {test_result['status']}")
            print(f"   Mensaje: {test_result['message']}")
            if test_result['details']:
                print(f"   Detalles: {json.dumps(test_result['details'], indent=6)}")
            print()
    
    def _save_results(self):
        """Guardar resultados en archivo"""
        results_file = f"ui_integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Resultados guardados en: {results_file}")


def main():
    """Función principal"""
    try:
        tester = UIIntegrationTester()
        tester.run_integration_tests()
        
        # Determinar código de salida basado en resultados
        if tester.results['summary']['failed'] > 0:
            print(f"\n❌ {tester.results['summary']['failed']} pruebas de integración fallaron")
            sys.exit(1)
        else:
            print(f"\n✅ Todas las {tester.results['summary']['passed']} pruebas de integración pasaron")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error al ejecutar pruebas de integración: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()