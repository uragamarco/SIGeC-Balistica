#!/usr/bin/env python3
"""
Prueba integral de la GUI con funcionalidades reales del backend
Incluye pruebas de carga de imágenes, procesamiento, base de datos y comparación
"""

import sys
import os
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np
import cv2
from datetime import datetime

# Configurar el path para importar módulos
sys.path.insert(0, os.path.abspath('.'))

# Configurar PyQt5 para modo headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest

# Importar módulos del sistema
from gui.main_window import MainWindow
from utils.config import get_config
from utils.logger import setup_logging
from image_processing.feature_extractor import extract_orb_features
from image_processing.preprocessor import BallisticPreprocessor
from database.vector_db import VectorDatabase
from matching.matcher import BallisticMatcher

class TestGUIComprehensive:
    """Pruebas integrales de la GUI con funcionalidades reales"""
    
    def __init__(self):
        self.app = None
        self.window = None
        self.temp_dir = None
        self.test_images = []
        self.config = None
        
    def setup(self):
        """Configuración inicial de las pruebas"""
        print("🔧 Configurando pruebas integrales de GUI...")
        
        # Crear aplicación PyQt5
        self.app = QApplication(sys.argv)
        
        # Cargar configuración
        self.config = get_config()
        setup_logging(
            log_level=self.config.logging.level,
            log_file=self.config.logging.file_path,
            console_output=True
        )
        
        # Crear directorio temporal para pruebas
        self.temp_dir = tempfile.mkdtemp(prefix="SIGeC-Balistica_gui_test_")
        print(f"📁 Directorio temporal: {self.temp_dir}")
        
        # Crear imágenes de prueba
        self._create_test_images()
        
        # Inicializar ventana principal
        self.window = MainWindow(self.config)
        
        print("✅ Configuración completada")
        
    def _create_test_images(self):
        """Crea imágenes de prueba sintéticas"""
        print("🖼️ Creando imágenes de prueba...")
        
        # Crear diferentes tipos de imágenes de prueba
        test_cases = [
            ("vaina_test_1.jpg", self._create_cartridge_case_image()),
            ("vaina_test_2.jpg", self._create_cartridge_case_image(noise_level=0.1)),
            ("proyectil_test_1.jpg", self._create_bullet_image()),
        ]
        
        for filename, image in test_cases:
            filepath = os.path.join(self.temp_dir, filename)
            cv2.imwrite(filepath, image)
            self.test_images.append(filepath)
            print(f"  ✓ Creada: {filename}")
    
    def _create_cartridge_case_image(self, noise_level=0.05):
        """Crea una imagen sintética de vaina"""
        # Crear imagen base de 512x512
        img = np.zeros((512, 512), dtype=np.uint8)
        
        # Círculo principal (culote de la vaina)
        cv2.circle(img, (256, 256), 200, 180, -1)
        
        # Círculo interno (percutor)
        cv2.circle(img, (256, 256), 30, 120, -1)
        
        # Añadir algunas marcas características
        for i in range(8):
            angle = i * 45
            x = int(256 + 150 * np.cos(np.radians(angle)))
            y = int(256 + 150 * np.sin(np.radians(angle)))
            cv2.circle(img, (x, y), 5, 220, -1)
        
        # Añadir ruido si se especifica
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        return img
    
    def _create_bullet_image(self):
        """Crea una imagen sintética de proyectil"""
        # Crear imagen base de 512x256 (proyectil alargado)
        img = np.zeros((256, 512), dtype=np.uint8)
        
        # Forma básica del proyectil
        cv2.rectangle(img, (50, 50), (462, 206), 160, -1)
        
        # Añadir estrías (líneas paralelas)
        for i in range(10, 246, 20):
            cv2.line(img, (60, i), (452, i), 200, 2)
        
        return img
    
    def test_gui_initialization(self):
        """Prueba la inicialización de la GUI"""
        print("\n🧪 Probando inicialización de GUI...")
        
        try:
            # Verificar que la ventana se creó correctamente
            assert self.window is not None, "Ventana principal no inicializada"
            
            # Verificar título
            expected_title = "SIGeC-Balistica- Sistema de Análisis Balístico v1.0"
            assert self.window.windowTitle() == expected_title, f"Título incorrecto: {self.window.windowTitle()}"
            
            # Verificar que las pestañas están presentes
            assert self.window.tab_widget.count() == 4, f"Número incorrecto de pestañas: {self.window.tab_widget.count()}"
            
            # Verificar nombres de pestañas
            expected_tabs = ["Cargar Imágenes", "Base de Datos", "Comparación", "Reportes"]
            for i, expected_name in enumerate(expected_tabs):
                actual_name = self.window.tab_widget.tabText(i)
                assert actual_name == expected_name, f"Pestaña {i}: esperado '{expected_name}', obtenido '{actual_name}'"
            
            print("  ✅ Inicialización de GUI correcta")
            return True
            
        except Exception as e:
            print(f"  ❌ Error en inicialización: {e}")
            return False
    
    def test_backend_integration(self):
        """Prueba la integración con el backend"""
        print("\n🧪 Probando integración con backend...")
        
        try:
            # Probar extractor de características
            test_image_path = self.test_images[0]
            
            # Cargar imagen en escala de grises
            test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
            assert test_image is not None, "No se pudo cargar la imagen de prueba"
            
            features = extract_orb_features(test_image)
            assert isinstance(features, dict), "Extractor no devuelve diccionario"
            assert 'num_keypoints' in features, "Faltan keypoints en features"
            
            print(f"  ✓ Extractor: {features.get('num_keypoints', 0)} keypoints detectados")
            
            # Probar preprocesador
            preprocessor = BallisticPreprocessor(self.config)
            processed_result = preprocessor.preprocess_ballistic_image(test_image_path)
            assert processed_result.success, "Preprocesador falló"
            
            print("  ✓ Preprocesador funcionando")
            
            # Probar base de datos vectorial
            vector_db = VectorDatabase(self.config)
            
            # Crear caso de prueba
            from database.vector_db import BallisticCase
            
            case_data = BallisticCase(
                case_number=f'TEST-{datetime.now().strftime("%Y%m%d%H%M%S")}',
                investigator='Test User',
                weapon_type='Pistola',
                caliber='9mm'
            )
            
            case_id = vector_db.add_case(case_data)
            assert case_id is not None, "No se pudo crear caso en BD"
            
            print(f"  ✓ Base de datos: caso {case_id} creado")
            
            # Probar matcher
            matcher = BallisticMatcher(self.config)
            
            # Extraer características de dos imágenes
            gray_image1 = cv2.imread(self.test_images[0], cv2.IMREAD_GRAYSCALE)
            gray_image2 = cv2.imread(self.test_images[1], cv2.IMREAD_GRAYSCALE)
            
            features1 = extract_orb_features(gray_image1)
            features2 = extract_orb_features(gray_image2)
            
            if features1.get('num_keypoints', 0) > 0 and features2.get('num_keypoints', 0) > 0:
                # Extraer características usando OpenCV directamente para el matcher
                orb = cv2.ORB_create(nfeatures=500)
                kp1, desc1 = orb.detectAndCompute(gray_image1, None)
                kp2, desc2 = orb.detectAndCompute(gray_image2, None)
                
                if desc1 is not None and desc2 is not None:
                    # Realizar matching
                    match_result = matcher.match_orb_features(desc1, desc2, kp1, kp2)
                    assert match_result.similarity_score >= 0, "Matching falló"
                    print("  ✓ Matcher: componentes inicializados")
                else:
                    print("  ⚠ Matcher: descriptores no válidos")
            else:
                print("  ⚠ Matcher: no hay suficientes keypoints para matching")
            
            print("  ✅ Integración con backend exitosa")
            return True
            
        except Exception as e:
            print(f"  ❌ Error en integración backend: {e}")
            return False
    
    def test_image_processing_workflow(self):
        """Prueba el flujo completo de procesamiento de imágenes"""
        print("\n🧪 Probando flujo de procesamiento de imágenes...")
        
        try:
            # Inicializar componentes
            preprocessor = BallisticPreprocessor(self.config)
            
            results = []
            
            for i, image_path in enumerate(self.test_images):
                print(f"  📷 Procesando imagen {i+1}: {os.path.basename(image_path)}")
                
                # Paso 1: Preprocesamiento
                processed_result = preprocessor.preprocess_ballistic_image(image_path)
                assert processed_result.success, f"Preprocesamiento falló para imagen {i+1}"
                
                # Paso 2: Extracción de características
                # Cargar imagen en escala de grises para extracción
                gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                assert gray_image is not None, f"No se pudo cargar imagen {i+1}"
                
                features = extract_orb_features(gray_image)
                assert isinstance(features, dict), f"Extracción falló para imagen {i+1}"
                
                num_keypoints = features.get('num_keypoints', 0)
                print(f"    ✓ {num_keypoints} keypoints extraídos")
                
                results.append({
                    'image': image_path,
                    'keypoints': num_keypoints,
                    'features': features
                })
            
            # Verificar que se procesaron todas las imágenes
            assert len(results) == len(self.test_images), "No se procesaron todas las imágenes"
            
            # Verificar que al menos una imagen tiene keypoints
            total_keypoints = sum(r['keypoints'] for r in results)
            assert total_keypoints > 0, "Ninguna imagen generó keypoints"
            
            print(f"  ✅ Flujo completado: {len(results)} imágenes, {total_keypoints} keypoints totales")
            return True
            
        except Exception as e:
            print(f"  ❌ Error en flujo de procesamiento: {e}")
            return False
    
    def test_database_operations(self):
        """Prueba operaciones de base de datos"""
        print("\n🧪 Probando operaciones de base de datos...")
        
        try:
            # Crear base de datos temporal
            db_path = os.path.join(self.temp_dir, "test_operations.sqlite")
            vector_db = VectorDatabase(self.config)
            
            # Crear múltiples casos
            from database.vector_db import BallisticCase, BallisticImage
            
            casos = [
                BallisticCase(
                    case_number=f"CASO-{datetime.now().strftime('%Y%m%d%H%M%S')}-001",
                    investigator="Inspector Test",
                    weapon_type="Pistola",
                    caliber="9mm"
                ),
                BallisticCase(
                    case_number=f"CASO-{datetime.now().strftime('%Y%m%d%H%M%S')}-002", 
                    investigator="Detective Prueba",
                    weapon_type="Revolver",
                    caliber=".38"
                )
            ]
            
            case_ids = []
            for caso in casos:
                case_id = vector_db.add_case(caso)
                assert case_id is not None, f"No se pudo crear caso {caso.case_number}"
                case_ids.append(case_id)
                print(f"  ✓ Caso creado: {caso.case_number} (ID: {case_id})")
            
            # Verificar que los casos se pueden recuperar
            casos_recuperados = vector_db.get_cases()
            assert len(casos_recuperados) >= 2, "No se pudieron recuperar los casos"
            print(f"  ✓ {len(casos_recuperados)} casos recuperados de BD")
            
            # Probar listado de casos
            todos_casos = vector_db.get_cases()
            assert len(todos_casos) >= 2, "No se encontraron suficientes casos"
            print(f"  ✓ Listado de casos: {len(todos_casos)} casos encontrados")
            
            print(f"  ✅ Operaciones de BD exitosas: {len(case_ids)} casos creados")
            return True
            
        except Exception as e:
            print(f"  ❌ Error en operaciones de BD: {e}")
            return False
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        print("🚀 Iniciando pruebas integrales de GUI...")
        
        tests = [
            ("Inicialización GUI", self.test_gui_initialization),
            ("Integración Backend", self.test_backend_integration),
            ("Flujo Procesamiento", self.test_image_processing_workflow),
            ("Operaciones BD", self.test_database_operations),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ Error crítico en {test_name}: {e}")
                results.append((test_name, False))
        
        # Resumen de resultados
        print("\n" + "="*60)
        print("📊 RESUMEN DE PRUEBAS INTEGRALES")
        print("="*60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            print(f"{status:10} | {test_name}")
            if result:
                passed += 1
        
        print("-"*60)
        print(f"Resultado: {passed}/{total} pruebas exitosas")
        
        if passed == total:
            print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        else:
            print(f"⚠️  {total - passed} pruebas fallaron")
        
        return passed == total
    
    def cleanup(self):
        """Limpieza después de las pruebas"""
        print("\n🧹 Limpiando recursos...")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"  ✓ Directorio temporal eliminado: {self.temp_dir}")
        
        if self.app:
            self.app.quit()
            print("  ✓ Aplicación PyQt5 cerrada")

def main():
    """Función principal"""
    test_suite = TestGUIComprehensive()
    
    try:
        test_suite.setup()
        success = test_suite.run_all_tests()
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error crítico en pruebas: {e}")
        return 1
        
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    sys.exit(main())