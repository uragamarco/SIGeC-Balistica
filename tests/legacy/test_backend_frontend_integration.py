#!/usr/bin/env python3
"""
Test de Integración Backend-Frontend
====================================

Prueba la integración completa entre los módulos de backend y frontend
usando datos reales de imágenes balísticas.

Autor: Sistema de Análisis Balístico SEACABAr
Fecha: 2024
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports del sistema
from config.unified_config import get_unified_config
from utils.logger import get_logger
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage

# Imports de procesamiento
try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
    from image_processing.feature_extractor import FeatureExtractor
    from matching.unified_matcher import UnifiedMatcher, MatchingConfig
    PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Módulos de procesamiento no disponibles: {e}")
    PROCESSING_AVAILABLE = False

# Imports de GUI
try:
    from PyQt5.QtWidgets import QApplication
    from gui.main_window import MainWindow
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: GUI no disponible: {e}")
    GUI_AVAILABLE = False

class BackendFrontendIntegrationTester:
    """Tester para integración backend-frontend"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_unified_config()
        self.temp_dir = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """Configura el entorno de pruebas"""
        print("🔧 Configurando entorno de pruebas...")
        
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="seacabar_test_")
        print(f"  📁 Directorio temporal: {self.temp_dir}")
        
        # Configurar base de datos temporal
        temp_db_path = os.path.join(self.temp_dir, "test_ballistic_database.db")
        self.config.database.sqlite_path = temp_db_path
        
        # Crear imágenes de prueba sintéticas
        self._create_test_images()
        
        print("  ✅ Entorno configurado")
        
    def _create_test_images(self):
        """Crea imágenes de prueba sintéticas"""
        import numpy as np
        from PIL import Image
        
        self.test_images_dir = os.path.join(self.temp_dir, "test_images")
        os.makedirs(self.test_images_dir, exist_ok=True)
        
        # Crear imágenes sintéticas de diferentes tipos
        test_cases = [
            ("vaina_001.jpg", "vaina", (800, 600)),
            ("vaina_002.jpg", "vaina", (800, 600)),
            ("proyectil_001.jpg", "proyectil", (600, 800)),
            ("proyectil_002.jpg", "proyectil", (600, 800))
        ]
        
        for filename, evidence_type, size in test_cases:
            # Crear imagen sintética con patrones
            img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            
            # Agregar algunos patrones para simular características balísticas
            if evidence_type == "vaina":
                # Simular marcas de percutor (círculo central)
                center_x, center_y = size[0] // 2, size[1] // 2
                y, x = np.ogrid[:size[1], :size[0]]
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= 50 ** 2
                img_array[mask] = [200, 200, 200]
            else:
                # Simular estrías (líneas verticales)
                for i in range(0, size[0], 20):
                    if i + 2 < size[0]:
                        img_array[:, i:i+2] = [150, 150, 150]
            
            # Guardar imagen
            img_path = os.path.join(self.test_images_dir, filename)
            Image.fromarray(img_array).save(img_path)
            
        print(f"  📸 Creadas {len(test_cases)} imágenes de prueba")
        
    def test_database_operations(self) -> bool:
        """Prueba operaciones de base de datos"""
        print("\n📦 Probando operaciones de base de datos...")
        
        try:
            # Inicializar base de datos
            db = VectorDatabase(self.config)
            
            # Crear caso de prueba
            test_case = BallisticCase(
                case_number="TEST_001",
                investigator="Test Investigator",
                date_created=datetime.now().isoformat(),
                weapon_type="Pistola",
                caliber="9mm",
                description="Caso de prueba para integración"
            )
            
            case_id = db.add_case(test_case)
            print(f"  ✅ Caso creado con ID: {case_id}")
            
            # Verificar que el caso se guardó
            cases = db.get_all_cases()
            assert len(cases) > 0, "No se encontraron casos en la base de datos"
            print(f"  ✅ Casos en BD: {len(cases)}")
            
            self.test_results['database'] = {
                'status': 'success',
                'case_id': case_id,
                'total_cases': len(cases)
            }
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error en operaciones de BD: {str(e)}")
            self.test_results['database'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_image_processing(self) -> bool:
        """Prueba el pipeline de procesamiento de imágenes"""
        print("\n🖼️ Probando procesamiento de imágenes...")
        
        if not PROCESSING_AVAILABLE:
            print("  ⚠️ Módulos de procesamiento no disponibles")
            self.test_results['image_processing'] = {
                'status': 'skipped',
                'reason': 'modules_not_available'
            }
            return False
        
        try:
            # Obtener imagen de prueba
            test_image = os.path.join(self.test_images_dir, "vaina_001.jpg")
            
            # Configurar preprocessor
            preprocessing_config = Preprocessingget_unified_config()
            preprocessor = UnifiedPreprocessor(preprocessing_config)
            
            # Procesar imagen
            processed_result = preprocessor.preprocess_ballistic_image(test_image, evidence_type="vaina")
            
            assert processed_result.success, f"Error en preprocesamiento: {processed_result.error_message}"
            assert processed_result.processed_image is not None, "Imagen procesada es None"
            
            print(f"  ✅ Imagen procesada exitosamente")
            print(f"  📊 Dimensiones: {processed_result.processed_image.shape}")
            
            # Extraer características
            extractor = FeatureExtractor()
            features = extractor.extract_orb_features(processed_result.processed_image)
            
            assert features is not None, "Extracción de características falló"
            assert 'keypoints' in features, "Features should contain keypoints"
            assert 'descriptors' in features, "Features should contain descriptors"
            print(f"  ✅ Características extraídas: {len(features.get('keypoints', []))} keypoints")
            
            self.test_results['image_processing'] = {
                'status': 'success',
                'keypoints_count': len(features.get('keypoints', [])),
                'image_shape': processed_result.processed_image.shape
            }
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error en procesamiento: {str(e)}")
            self.test_results['image_processing'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_matching_system(self) -> bool:
        """Prueba el sistema de matching"""
        print("\n🔍 Probando sistema de matching...")
        
        if not PROCESSING_AVAILABLE:
            print("  ⚠️ Sistema de matching no disponible")
            self.test_results['matching'] = {
                'status': 'skipped',
                'reason': 'modules_not_available'
            }
            return False
        
        try:
            # Configurar matcher
            matching_config = Matchingget_unified_config()
            matcher = UnifiedMatcher(self.config)
            
            # Obtener imágenes de prueba
            image1 = os.path.join(self.test_images_dir, "vaina_001.jpg")
            image2 = os.path.join(self.test_images_dir, "vaina_002.jpg")
            
            # Realizar comparación
            match_result = matcher.compare_image_files(image1, image2)
            
            assert match_result is not None, "Matching devolvió None"
            assert hasattr(match_result, 'similarity_score'), "Falta score de similitud"
            
            print(f"  ✅ Comparación exitosa")
            print(f"  📊 Score de similitud: {match_result.similarity_score:.3f}")
            print(f"  🔗 Matches encontrados: {len(match_result.matches)}")
            
            self.test_results['matching'] = {
                'status': 'success',
                'similarity_score': float(match_result.similarity_score),
                'matches_count': len(match_result.matches)
            }
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error en matching: {str(e)}")
            self.test_results['matching'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_gui_initialization(self) -> bool:
        """Prueba la inicialización de la GUI"""
        print("\n🖥️ Probando inicialización de GUI...")
        
        if not GUI_AVAILABLE:
            print("  ⚠️ GUI no disponible")
            self.test_results['gui'] = {
                'status': 'skipped',
                'reason': 'gui_not_available'
            }
            return False
        
        try:
            # Configurar Qt para modo headless
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            
            # Crear aplicación Qt (sin mostrar)
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            
            # Inicializar ventana principal
            main_window = MainWindow(self.config)
            
            # Verificar componentes principales
            assert hasattr(main_window, 'tab_widget'), "Falta tab_widget"
            assert hasattr(main_window, 'config'), "Falta config"
            
            print(f"  ✅ Ventana principal inicializada")
            print(f"  📑 Tabs disponibles: {main_window.tab_widget.count()}")
            
            self.test_results['gui'] = {
                'status': 'success',
                'tabs_count': main_window.tab_widget.count()
            }
            
            # Limpiar
            main_window.close()
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error en GUI: {str(e)}")
            self.test_results['gui'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_end_to_end_workflow(self) -> bool:
        """Prueba el flujo completo end-to-end"""
        print("\n🔄 Probando flujo completo end-to-end...")
        
        if not PROCESSING_AVAILABLE:
            print("  ⚠️ Flujo completo no disponible sin módulos de procesamiento")
            self.test_results['end_to_end'] = {
                'status': 'skipped',
                'reason': 'processing_not_available'
            }
            return False
        
        try:
            # 1. Inicializar componentes
            db = VectorDatabase(self.config)
            matcher = UnifiedMatcher(self.config)
            
            # 2. Crear caso
            test_case = BallisticCase(
                case_number="E2E_TEST",
                investigator="Integration Tester",
                date_created=datetime.now().isoformat(),
                weapon_type="Pistola",
                caliber="9mm"
            )
            case_id = db.add_case(test_case)
            
            # 3. Procesar y almacenar imagen
            test_image = os.path.join(self.test_images_dir, "vaina_001.jpg")
            
            # Simular procesamiento completo
            processed_result = matcher.preprocessor.preprocess_ballistic_image(test_image, evidence_type="vaina")
            features = matcher.feature_extractor.extract_orb_features(processed_result.processed_image)
            
            # 4. Guardar en base de datos
            ballistic_image = BallisticImage(
                case_id=case_id,
                image_path=test_image,
                evidence_type="vaina",
                processing_metadata=processed_result.metadata or {},
                feature_vector=features.get('descriptors', [])
            )
            
            image_id = db.add_image(ballistic_image)
            
            # 5. Realizar búsqueda
            search_results = db.search_similar_images(
                query_vector=features.get('descriptors', []), 
                top_k=5
            )
            
            print(f"  ✅ Flujo completo exitoso")
            print(f"  📁 Caso ID: {case_id}")
            print(f"  🖼️ Imagen ID: {image_id}")
            print(f"  🔍 Resultados de búsqueda: {len(search_results)}")
            
            self.test_results['end_to_end'] = {
                'status': 'success',
                'case_id': case_id,
                'image_id': image_id,
                'search_results_count': len(search_results)
            }
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error en flujo E2E: {str(e)}")
            self.test_results['end_to_end'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def cleanup_test_environment(self):
        """Limpia el entorno de pruebas"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Limpieza completada: {self.temp_dir}")
    
    def generate_report(self):
        """Genera reporte de resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"backend_frontend_integration_results_{timestamp}.json"
        
        # Calcular estadísticas
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get('status') == 'success')
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'error')
        skipped_tests = sum(1 for result in self.test_results.values() 
                           if result.get('status') == 'skipped')
        
        # Determinar estado general
        if failed_tests == 0 and successful_tests > 0:
            overall_status = "✅ EXCELENTE"
        elif failed_tests <= 1 and successful_tests >= 3:
            overall_status = "⚠️ ACEPTABLE"
        else:
            overall_status = "❌ REQUIERE ATENCIÓN"
        
        report_data = {
            'timestamp': timestamp,
            'overall_status': overall_status,
            'summary': {
                'total_tests': total_tests,
                'successful': successful_tests,
                'failed': failed_tests,
                'skipped': skipped_tests
            },
            'detailed_results': self.test_results
        }
        
        # Guardar reporte
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n============================================================")
        print(f"📊 RESUMEN DE INTEGRACIÓN BACKEND-FRONTEND")
        print(f"============================================================")
        print(f"✅ Pruebas exitosas: {successful_tests}/{total_tests}")
        print(f"❌ Pruebas fallidas: {failed_tests}/{total_tests}")
        print(f"⚠️ Pruebas omitidas: {skipped_tests}/{total_tests}")
        print(f"\n📄 Reporte guardado en: {report_file}")
        print(f"🎯 Estado general: {overall_status}")
        
        return overall_status

def main():
    """Función principal"""
    print("🚀 Iniciando pruebas de integración Backend-Frontend")
    print("=" * 60)
    
    tester = BackendFrontendIntegrationTester()
    
    try:
        # Configurar entorno
        tester.setup_test_environment()
        
        # Ejecutar pruebas
        tests = [
            tester.test_database_operations,
            tester.test_image_processing,
            tester.test_matching_system,
            tester.test_gui_initialization,
            tester.test_end_to_end_workflow
        ]
        
        for test in tests:
            test()
        
        # Generar reporte
        overall_status = tester.generate_report()
        
        # Determinar código de salida
        if "EXCELENTE" in overall_status:
            exit_code = 0
        elif "ACEPTABLE" in overall_status:
            exit_code = 0
        else:
            exit_code = 1
        
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Error crítico en las pruebas: {str(e)}")
        return 1
        
    finally:
        tester.cleanup_test_environment()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)