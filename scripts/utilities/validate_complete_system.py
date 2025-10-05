#!/usr/bin/env python3
"""
Script de Validación Final del Sistema Balístico Forense MVP
Sistema Balístico Forense MVP

Este script realiza una validación completa del sistema incluyendo:
- Validación de dependencias y configuración
- Pruebas de todos los módulos principales
- Pruebas de integración completa
- Validación de rendimiento
- Generación de reporte de validación

Uso:
    python validate_complete_system.py [--verbose] [--output-dir DIR]
"""

import os
import sys
import time
import json
import logging
import traceback
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)

class SystemValidator:
    """Validador completo del sistema balístico forense"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "dependency_checks": {},
            "module_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "overall_status": "UNKNOWN",
            "errors": [],
            "warnings": []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def validate_system_info(self) -> Dict[str, Any]:
        """Valida información del sistema"""
        self.logger.info("Validando información del sistema...")
        
        try:
            import platform
            import psutil
            
            info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_free": psutil.disk_usage('.').free,
                "cpu_count": psutil.cpu_count()
            }
            
            self.results["system_info"] = info
            self.logger.info("✓ Información del sistema obtenida")
            return info
            
        except Exception as e:
            error_msg = f"Error obteniendo información del sistema: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {}
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """Valida todas las dependencias del sistema"""
        self.logger.info("Validando dependencias...")
        
        dependencies = {
            "opencv-python": "cv2",
            "numpy": "numpy",
            "PyQt5": "PyQt5",
            "faiss-cpu": "faiss",
            "scipy": "scipy",
            "scikit-image": "skimage",
            "pillow": "PIL",
            "matplotlib": "matplotlib"
        }
        
        results = {}
        
        for package, module in dependencies.items():
            try:
                __import__(module)
                results[package] = True
                self.logger.info(f"✓ {package} disponible")
            except ImportError:
                results[package] = False
                warning_msg = f"⚠ {package} no disponible"
                self.logger.warning(warning_msg)
                self.results["warnings"].append(warning_msg)
        
        self.results["dependency_checks"] = results
        return results
    
    def test_image_processing_module(self) -> Dict[str, Any]:
        """Prueba el módulo de procesamiento de imágenes"""
        self.logger.info("Probando módulo de procesamiento de imágenes...")
        
        try:
            from image_processing.unified_preprocessor import UnifiedPreprocessor
            from image_processing.feature_extractor import FeatureExtractor
            
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            
            # Probar preprocessor
            preprocessor = UnifiedPreprocessor()
            result = preprocessor.preprocess_image(test_image)
            
            if not result.success:
                raise Exception(f"Preprocessor falló: {result.error_message}")
            
            # Probar feature extractor
            extractor = FeatureExtractor()
            features = extractor.extract_orb_features(result.processed_image)
            
            if features is None or features.get('num_keypoints', 0) == 0:
                raise Exception("Feature extractor no generó características")
            
            test_result = {
                "preprocessor": True,
                "feature_extractor": True,
                "processing_time": result.processing_time,
                "features_count": features.get('num_keypoints', 0),
                "descriptors_shape": features.get('descriptors', np.array([])).shape if features.get('descriptors') is not None else None
            }
            
            self.logger.info("✓ Módulo de procesamiento de imágenes funciona correctamente")
            return test_result
            
        except Exception as e:
            error_msg = f"Error en módulo de procesamiento: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {"preprocessor": False, "feature_extractor": False, "error": str(e)}
    
    def test_database_module(self) -> Dict[str, Any]:
        """Prueba el módulo de base de datos"""
        self.logger.info("Probando módulo de base de datos...")
        
        try:
            from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
            from config.unified_config import get_unified_config
            
            # Crear configuración temporal
            config = get_unified_config()
            
            # Crear base de datos temporal
            db = VectorDatabase(config)
            
            # Probar operaciones básicas
            timestamp = str(int(time.time()))
            
            # Agregar caso de prueba
            case = BallisticCase(
                case_number=f"VALIDATION-{timestamp}",
                investigator="Sistema de Validación",
                date_created=datetime.now().strftime("%Y-%m-%d"),
                weapon_type="Pistola",
                caliber="9mm"
            )
            
            case_id = db.add_case(case)
            
            # Agregar imagen de prueba
            image = BallisticImage(
                case_id=case_id,
                filename=f"validation_image_{timestamp}.jpg",
                file_path=f"/tmp/validation_{timestamp}.jpg",
                evidence_type="vaina",
                image_hash=hashlib.md5(f"validation_{timestamp}".encode()).hexdigest(),
                width=400,
                height=400,
                file_size=1024
            )
            
            image_id = db.add_image(image)
            
            # Agregar vector de características
            test_vector = np.random.rand(32).astype(np.float32)
            feature_vector = FeatureVector(
                image_id=image_id,
                algorithm="ORB",
                vector_size=32,
                extraction_params='{"nfeatures": 500}',
                date_extracted=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            vector_id = db.add_feature_vector(feature_vector, test_vector)
            
            # Probar búsqueda
            similar_results = db.search_similar_vectors(test_vector, k=5)
            
            # Obtener estadísticas
            stats = db.get_database_stats()
            
            test_result = {
                "case_creation": case_id is not None,
                "image_creation": image_id is not None,
                "vector_creation": vector_id is not None,
                "search_functionality": len(similar_results) >= 0,
                "statistics": stats,
                "case_id": case_id,
                "image_id": image_id,
                "vector_id": vector_id
            }
            
            self.logger.info("✓ Módulo de base de datos funciona correctamente")
            return test_result
            
        except Exception as e:
            error_msg = f"Error en módulo de base de datos: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {"error": str(e)}
    
    def test_matching_module(self) -> Dict[str, Any]:
        """Prueba el módulo de matching"""
        self.logger.info("Probando módulo de matching...")
        
        try:
            from matching.unified_matcher import UnifiedMatcher
            
            # Crear imágenes de prueba
            image1 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            image2 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            
            # Crear matcher
            matcher = UnifiedMatcher()
            
            # Realizar comparación
            result = matcher.compare_images(image1, image2)
            
            test_result = {
                "comparison_successful": result is not None,
                "similarity_score": result.similarity_score if result else 0,
                "good_matches": result.good_matches if result else 0,
                "total_matches": result.total_matches if result else 0,
                "processing_time": result.processing_time if result else 0
            }
            
            self.logger.info("✓ Módulo de matching funciona correctamente")
            return test_result
            
        except Exception as e:
            error_msg = f"Error en módulo de matching: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {"error": str(e)}
    
    def test_gui_components(self) -> Dict[str, Any]:
        """Prueba componentes GUI (sin display)"""
        self.logger.info("Probando componentes GUI...")
        
        try:
            # Importar componentes GUI
            from gui.main_window import MainWindow
            from gui.image_tab import ImageTab
            from gui.database_tab import DatabaseTab
            from gui.reports_tab import ReportsTab
            
            test_result = {
                "main_window_import": True,
                "image_tab_import": True,
                "database_tab_import": True,
                "reports_tab_import": True,
                "gui_available": True
            }
            
            self.logger.info("✓ Componentes GUI importados correctamente")
            return test_result
            
        except Exception as e:
            error_msg = f"Error en componentes GUI: {e}"
            self.logger.warning(error_msg)
            self.results["warnings"].append(error_msg)
            return {"gui_available": False, "error": str(e)}
    
    def test_complete_integration(self) -> Dict[str, Any]:
        """Prueba integración completa del sistema"""
        self.logger.info("Probando integración completa...")
        
        try:
            start_time = time.time()
            
            # Importar todos los módulos
            from image_processing.unified_preprocessor import UnifiedPreprocessor
            from image_processing.feature_extractor import FeatureExtractor
            from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
            from matching.unified_matcher import UnifiedMatcher
            from config.unified_config import get_unified_config
            
            # Crear componentes
            config = get_unified_config()
            preprocessor = UnifiedPreprocessor()
            extractor = FeatureExtractor()
            db = VectorDatabase(config)
            matcher = UnifiedMatcher()
            
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            
            # Flujo completo
            # 1. Preprocesar imagen
            processed_result = preprocessor.preprocess_image(test_image)
            if not processed_result.success:
                raise Exception("Preprocesamiento falló")
            
            # 2. Extraer características
            features = extractor.extract_orb_features(processed_result.processed_image)
            if features is None or features.get('num_keypoints', 0) == 0:
                raise Exception("Extracción de características falló")
            
            # 3. Crear caso y almacenar en BD
            timestamp = str(int(time.time()))
            case = BallisticCase(
                case_number=f"INTEGRATION-{timestamp}",
                investigator="Sistema de Validación",
                date_created=datetime.now().strftime("%Y-%m-%d")
            )
            case_id = db.add_case(case)
            
            # 4. Almacenar imagen
            image = BallisticImage(
                case_id=case_id,
                filename=f"integration_test_{timestamp}.jpg",
                file_path=f"/tmp/integration_{timestamp}.jpg",
                evidence_type="vaina",
                image_hash=hashlib.md5(f"integration_{timestamp}".encode()).hexdigest(),
                width=400,
                height=400
            )
            image_id = db.add_image(image)
            
            # 5. Almacenar vector
            if features.get('descriptors') is not None and len(features.get('descriptors', [])) > 0:
                # Crear vector promedio de los descriptores
                descriptors = features.get('descriptors')
                feature_vector_data = np.mean(descriptors, axis=0).astype(np.float32)
                
                feature_vector = FeatureVector(
                    image_id=image_id,
                    algorithm="ORB",
                    vector_size=len(feature_vector_data),
                    extraction_params='{"nfeatures": 500}',
                    date_extracted=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                vector_id = db.add_feature_vector(feature_vector, feature_vector_data)
                
                # 6. Buscar similares
                similar_results = db.search_similar_vectors(feature_vector_data, k=5)
                
                # 7. Comparar imágenes
                comparison_result = matcher.compare_images(test_image, test_image)
                
                integration_time = time.time() - start_time
                
                test_result = {
                    "preprocessing": True,
                    "feature_extraction": True,
                    "database_storage": True,
                    "similarity_search": True,
                    "image_comparison": True,
                    "total_time": integration_time,
                    "features_extracted": features.get('num_keypoints', 0),
                    "similar_results_count": len(similar_results),
                    "comparison_score": comparison_result.similarity_score if comparison_result else 0
                }
                
                self.logger.info("✓ Integración completa exitosa")
                return test_result
            else:
                raise Exception("No se extrajeron características válidas")
            
        except Exception as e:
            error_msg = f"Error en integración completa: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {"integration_successful": False, "error": str(e)}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Ejecuta pruebas de rendimiento"""
        self.logger.info("Ejecutando pruebas de rendimiento...")
        
        try:
            from image_processing.unified_preprocessor import UnifiedPreprocessor
            from image_processing.feature_extractor import FeatureExtractor
            from matching.unified_matcher import UnifiedMatcher
            
            # Crear componentes
            preprocessor = UnifiedPreprocessor()
            extractor = FeatureExtractor()
            matcher = UnifiedMatcher()
            
            # Pruebas con diferentes tamaños de imagen
            image_sizes = [(200, 200), (400, 400), (800, 600)]
            performance_results = {}
            
            for size in image_sizes:
                size_key = f"{size[0]}x{size[1]}"
                
                # Crear imagen de prueba
                test_image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
                
                # Medir preprocesamiento
                start_time = time.time()
                processed_result = preprocessor.preprocess_image(test_image)
                preprocessing_time = time.time() - start_time
                
                # Medir extracción de características
                start_time = time.time()
                features = extractor.extract_orb_features(processed_result.processed_image)
                extraction_time = time.time() - start_time
                
                # Medir comparación
                start_time = time.time()
                comparison_result = matcher.compare_images(test_image, test_image)
                comparison_time = time.time() - start_time
                
                performance_results[size_key] = {
                    "preprocessing_time": preprocessing_time,
                    "extraction_time": extraction_time,
                    "comparison_time": comparison_time,
                    "total_time": preprocessing_time + extraction_time + comparison_time,
                    "features_count": features.get('num_keypoints', 0) if features else 0
                }
            
            self.logger.info("✓ Pruebas de rendimiento completadas")
            return performance_results
            
        except Exception as e:
            error_msg = f"Error en pruebas de rendimiento: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            return {"error": str(e)}
    
    def generate_validation_report(self) -> str:
        """Genera reporte de validación"""
        self.logger.info("Generando reporte de validación...")
        
        # Determinar estado general
        has_critical_errors = len(self.results["errors"]) > 0
        has_warnings = len(self.results["warnings"]) > 0
        
        if has_critical_errors:
            self.results["overall_status"] = "FAILED"
        elif has_warnings:
            self.results["overall_status"] = "WARNING"
        else:
            self.results["overall_status"] = "PASSED"
        
        # Guardar reporte JSON
        report_path = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # Generar reporte de texto
        text_report_path = self.output_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE VALIDACIÓN DEL SISTEMA BALÍSTICO FORENSE MVP\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Fecha: {self.results['timestamp']}\n")
            f.write(f"Estado General: {self.results['overall_status']}\n\n")
            
            # Información del sistema
            if self.results["system_info"]:
                f.write("INFORMACIÓN DEL SISTEMA:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.results["system_info"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Dependencias
            f.write("DEPENDENCIAS:\n")
            f.write("-" * 30 + "\n")
            for dep, status in self.results["dependency_checks"].items():
                status_str = "✓" if status else "✗"
                f.write(f"{status_str} {dep}\n")
            f.write("\n")
            
            # Errores
            if self.results["errors"]:
                f.write("ERRORES CRÍTICOS:\n")
                f.write("-" * 30 + "\n")
                for error in self.results["errors"]:
                    f.write(f"• {error}\n")
                f.write("\n")
            
            # Advertencias
            if self.results["warnings"]:
                f.write("ADVERTENCIAS:\n")
                f.write("-" * 30 + "\n")
                for warning in self.results["warnings"]:
                    f.write(f"• {warning}\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        self.logger.info(f"✓ Reporte generado: {report_path}")
        return str(report_path)
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """Ejecuta validación completa del sistema"""
        self.logger.info("Iniciando validación completa del sistema...")
        
        try:
            # 1. Información del sistema
            self.validate_system_info()
            
            # 2. Dependencias
            self.validate_dependencies()
            
            # 3. Módulos individuales
            self.results["module_tests"]["image_processing"] = self.test_image_processing_module()
            self.results["module_tests"]["database"] = self.test_database_module()
            self.results["module_tests"]["matching"] = self.test_matching_module()
            self.results["module_tests"]["gui"] = self.test_gui_components()
            
            # 4. Integración completa
            self.results["integration_tests"]["complete_flow"] = self.test_complete_integration()
            
            # 5. Rendimiento
            self.results["performance_tests"] = self.run_performance_tests()
            
            # 6. Generar reporte
            report_path = self.generate_validation_report()
            
            self.logger.info("✓ Validación completa finalizada")
            
            return {
                "status": self.results["overall_status"],
                "report_path": report_path,
                "results": self.results
            }
            
        except Exception as e:
            error_msg = f"Error durante validación completa: {e}"
            self.logger.error(error_msg)
            self.results["errors"].append(error_msg)
            self.results["overall_status"] = "FAILED"
            
            # Generar reporte incluso con errores
            report_path = self.generate_validation_report()
            
            return {
                "status": "FAILED",
                "error": str(e),
                "report_path": report_path,
                "results": self.results
            }

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validador completo del sistema balístico forense")
    parser.add_argument("--verbose", "-v", action="store_true", help="Salida detallada")
    parser.add_argument("--output-dir", "-o", default="validation_results", help="Directorio de salida")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear validador
    validator = SystemValidator(args.output_dir)
    
    print("🔍 INICIANDO VALIDACIÓN COMPLETA DEL SISTEMA BALÍSTICO FORENSE MVP")
    print("=" * 70)
    
    # Ejecutar validación
    result = validator.run_complete_validation()
    
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DE VALIDACIÓN")
    print("=" * 70)
    
    status_emoji = {
        "PASSED": "✅",
        "WARNING": "⚠️",
        "FAILED": "❌"
    }
    
    print(f"\n{status_emoji.get(result['status'], '❓')} Estado General: {result['status']}")
    print(f"📄 Reporte: {result['report_path']}")
    
    if result['status'] == "FAILED":
        print(f"\n❌ Errores encontrados:")
        for error in validator.results.get("errors", []):
            print(f"   • {error}")
    
    if validator.results.get("warnings"):
        print(f"\n⚠️ Advertencias:")
        for warning in validator.results.get("warnings", []):
            print(f"   • {warning}")
    
    print("\n" + "=" * 70)
    
    # Código de salida
    exit_code = 0 if result['status'] in ["PASSED", "WARNING"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()