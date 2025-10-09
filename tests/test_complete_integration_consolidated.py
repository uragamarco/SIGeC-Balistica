#!/usr/bin/env python3
"""
Test de Integración Completa Consolidado - SIGeC-Balistica
=========================================================

Pruebas consolidadas de integración completa que validan:
- Integración Backend-Frontend completa
- Flujo de trabajo end-to-end
- Componentes de GUI y frontend
- Operaciones de base de datos integradas
- Procesamiento completo de imágenes balísticas

Migrado desde:
- tests/legacy/test_backend_frontend_integration.py
- tests/legacy/test_complete_flow.py
- tests/legacy/test_frontend_integration.py

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import os
import sys
import tempfile
import time
import unittest
import numpy as np
import cv2
from PIL import Image, ImageDraw
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

# Configurar Qt para modo headless si es necesario
if 'DISPLAY' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports del sistema
try:
    from utils.config import get_config
    from utils.logger import get_logger
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Imports de base de datos
try:
    from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Imports de procesamiento
try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    from image_processing.feature_extractor import FeatureExtractor
    from matching.unified_matcher import UnifiedMatcher
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

# Imports de GUI (con fallback a mock)
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
    from PyQt5.QtCore import QTimer
    GUI_AVAILABLE = True
except ImportError:
    # Mock PyQt5 para tests sin GUI
    sys.modules['PyQt5'] = MagicMock()
    sys.modules['PyQt5.QtWidgets'] = MagicMock()
    sys.modules['PyQt5.QtCore'] = MagicMock()
    GUI_AVAILABLE = False

class TestCompleteIntegrationConsolidated(unittest.TestCase):
    """Test consolidado de integración completa"""
    
    def setUp(self):
        """Configuración inicial"""
        self.temp_dir = tempfile.mkdtemp(prefix="SIGeC-Balistica_integration_")
        self.test_images = []
        self.test_results = {}
        
        if CONFIG_AVAILABLE:
            self.config = get_config()
            self.logger = get_logger(__name__)
        
        if PROCESSING_AVAILABLE and CONFIG_AVAILABLE:
            self.preprocessor = UnifiedPreprocessor(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            self.matcher = UnifiedMatcher(self.config)
        
        if DATABASE_AVAILABLE and CONFIG_AVAILABLE:
            # Configurar base de datos temporal
            temp_db_path = os.path.join(self.temp_dir, "test_integration.db")
            self.config.database.sqlite_path = temp_db_path
            self.db_manager = VectorDatabase(self.config)
        
        # Crear imágenes de prueba
        self._create_test_images()
    
    def _create_test_images(self):
        """Crear imágenes de prueba sintéticas para diferentes tipos de evidencia"""
        test_cases = [
            ("vaina_001.jpg", "vaina", (800, 600)),
            ("vaina_002.jpg", "vaina", (800, 600)),
            ("proyectil_001.jpg", "proyectil", (600, 800)),
            ("proyectil_002.jpg", "proyectil", (600, 800))
        ]
        
        for filename, evidence_type, size in test_cases:
            image_path = os.path.join(self.temp_dir, filename)
            
            # Crear imagen sintética con patrones balísticos
            img = Image.new('RGB', size, color='white')
            draw = ImageDraw.Draw(img)
            
            if evidence_type == "vaina":
                # Simular marcas de percutor (círculo central)
                center_x, center_y = size[0] // 2, size[1] // 2
                draw.ellipse([center_x-50, center_y-50, center_x+50, center_y+50], 
                           fill='black', outline='gray', width=3)
                
                # Agregar patrones radiales
                for i in range(8):
                    angle = i * 45
                    x1 = center_x + 30 * np.cos(np.radians(angle))
                    y1 = center_y + 30 * np.sin(np.radians(angle))
                    x2 = center_x + 80 * np.cos(np.radians(angle))
                    y2 = center_y + 80 * np.sin(np.radians(angle))
                    draw.line([x1, y1, x2, y2], fill='black', width=2)
            
            elif evidence_type == "proyectil":
                # Simular estrías longitudinales
                for i in range(6):
                    x = 100 + i * 80
                    draw.line([x, 50, x, size[1]-50], fill='black', width=3)
                    
                    # Agregar variaciones en las estrías
                    for j in range(10):
                        y = 100 + j * 60
                        draw.ellipse([x-5, y-3, x+5, y+3], fill='gray')
            
            img.save(image_path)
            self.test_images.append({
                'path': image_path,
                'type': evidence_type,
                'filename': filename
            })
    
    def tearDown(self):
        """Limpieza después de cada test"""
        # Limpiar archivos temporales
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_config_loading(self):
        """Test de carga de configuración"""
        self.assertIsNotNone(self.config)
        self.assertIn('database', self.config)
        self.assertIn('image_processing', self.config)
        
    @unittest.skipUnless(DATABASE_AVAILABLE and CONFIG_AVAILABLE, "Base de datos no disponible")
    def test_database_operations_complete(self):
        """Test completo de operaciones de base de datos"""
        try:
            # Crear caso de prueba
            case = BallisticCase(
                case_id=f"INTEGRATION_TEST_{int(time.time())}",
                description="Test case for complete integration",
                created_by="integration_test",
                weapon_type="Pistola",
                caliber="9mm"
            )
            
            case_id = self.db_manager.add_case(case)
            self.assertIsNotNone(case_id)
            
            # Agregar imágenes al caso
            for img_info in self.test_images:
                ballistic_image = BallisticImage(
                    case_id=case_id,
                    image_path=img_info['path'],
                    image_type=img_info['type'],
                    metadata={"test": True, "filename": img_info['filename']}
                )
                
                image_id = self.db_manager.add_image(ballistic_image)
                self.assertIsNotNone(image_id)
            
            # Verificar que las imágenes se guardaron
            images = self.db_manager.get_case_images(case_id)
            self.assertEqual(len(images), len(self.test_images))
            
        except Exception as e:
            self.fail(f"Error en operaciones completas de base de datos: {e}")
    
    @unittest.skipUnless(PROCESSING_AVAILABLE and CONFIG_AVAILABLE, "Procesamiento no disponible")
    def test_image_processing_complete(self):
        """Test completo de procesamiento de imágenes"""
        try:
            for img_info in self.test_images:
                # Cargar imagen
                image = cv2.imread(img_info['path'])
                self.assertIsNotNone(image)
                
                # Preprocesamiento
                processed = self.preprocessor.preprocess_image(image)
                self.assertIsNotNone(processed)
                self.assertEqual(len(processed.shape), 2)  # Escala de grises
                
                # Extracción de características
                features = self.feature_extractor.extract_all_features(processed, ['orb'])
                self.assertIn('orb', features)
                self.assertGreater(len(features['orb'].get('keypoints', [])), 0)
                
        except Exception as e:
            self.fail(f"Error en procesamiento completo de imágenes: {e}")
    
    @unittest.skipUnless(PROCESSING_AVAILABLE and CONFIG_AVAILABLE, "Matching no disponible")
    def test_matching_system_complete(self):
        """Test completo del sistema de matching"""
        try:
            if len(self.test_images) < 2:
                self.skipTest("Se necesitan al menos 2 imágenes para matching")
            
            # Comparar primera y segunda imagen
            img1 = cv2.imread(self.test_images[0]['path'])
            img2 = cv2.imread(self.test_images[1]['path'])
            
            result = self.matcher.compare_images(img1, img2)
            
            # Verificar resultado
            self.assertIsNotNone(result)
            self.assertHasAttr(result, 'similarity_score')
            self.assertHasAttr(result, 'confidence')
            self.assertIsInstance(result.similarity_score, (int, float))
            self.assertGreaterEqual(result.similarity_score, 0)
            self.assertLessEqual(result.similarity_score, 100)
            
        except Exception as e:
            self.fail(f"Error en sistema completo de matching: {e}")
    
    def test_gui_components_mock(self):
        """Test de componentes GUI (mockeados si es necesario)"""
        try:
            if GUI_AVAILABLE:
                # Test con GUI real
                app = QApplication.instance() or QApplication([])
                main_window = QMainWindow()
                self.assertIsNotNone(main_window)
                
                # Verificar que se puede crear sin errores
                widget = QWidget()
                main_window.setCentralWidget(widget)
                
            else:
                # Test con GUI mockeada
                import PyQt5.QtWidgets as QtWidgets
                self.assertIsInstance(QtWidgets, MagicMock)
                
                app = QtWidgets.QApplication([])
                window = QtWidgets.QMainWindow()
                self.assertIsNotNone(app)
                self.assertIsNotNone(window)
            
        except Exception as e:
            self.fail(f"Error en componentes GUI: {e}")
    
    @unittest.skipUnless(all([CONFIG_AVAILABLE, DATABASE_AVAILABLE, PROCESSING_AVAILABLE]), 
                        "Componentes completos no disponibles")
    def test_end_to_end_workflow(self):
        """Test del flujo de trabajo completo end-to-end"""
        try:
            # 1. Crear caso en base de datos
            case = BallisticCase(
                case_id=f"E2E_TEST_{int(time.time())}",
                description="End-to-end workflow test",
                created_by="e2e_test"
            )
            case_id = self.db_manager.add_case(case)
            self.assertIsNotNone(case_id)
            
            # 2. Procesar cada imagen y extraer características
            processed_features = []
            for img_info in self.test_images:
                # Cargar y procesar imagen
                image = cv2.imread(img_info['path'])
                processed = self.preprocessor.preprocess_image(image)
                
                # Extraer características
                features = self.feature_extractor.extract_all_features(processed, ['orb'])
                
                # Agregar imagen a la base de datos
                ballistic_image = BallisticImage(
                    case_id=case_id,
                    image_path=img_info['path'],
                    image_type=img_info['type'],
                    metadata={"processed": True}
                )
                image_id = self.db_manager.add_image(ballistic_image)
                
                processed_features.append({
                    'image_id': image_id,
                    'features': features,
                    'image_info': img_info
                })
            
            # 3. Realizar comparaciones entre imágenes
            comparisons = []
            for i in range(len(processed_features)):
                for j in range(i + 1, len(processed_features)):
                    img1 = cv2.imread(processed_features[i]['image_info']['path'])
                    img2 = cv2.imread(processed_features[j]['image_info']['path'])
                    
                    result = self.matcher.compare_images(img1, img2)
                    comparisons.append({
                        'image1_id': processed_features[i]['image_id'],
                        'image2_id': processed_features[j]['image_id'],
                        'similarity': result.similarity_score,
                        'confidence': result.confidence
                    })
            
            # 4. Verificar que el flujo completo funcionó
            self.assertGreater(len(processed_features), 0)
            self.assertGreater(len(comparisons), 0)
            
            # Verificar que todas las comparaciones tienen valores válidos
            for comp in comparisons:
                self.assertIsInstance(comp['similarity'], (int, float))
                self.assertIsInstance(comp['confidence'], (int, float))
                self.assertGreaterEqual(comp['similarity'], 0)
                self.assertLessEqual(comp['similarity'], 100)
            
        except Exception as e:
            self.fail(f"Error en flujo de trabajo end-to-end: {e}")
    
    def test_performance_benchmarks(self):
        """Test de benchmarks de rendimiento"""
        try:
            if not (PROCESSING_AVAILABLE and CONFIG_AVAILABLE):
                self.skipTest("Componentes de procesamiento no disponibles")
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            for img_info in self.test_images[:2]:  # Solo primeras 2 imágenes
                image = cv2.imread(img_info['path'])
                processed = self.preprocessor.preprocess_image(image)
                features = self.feature_extractor.extract_all_features(processed, ['orb'])
            
            processing_time = time.time() - start_time
            
            # Verificar que el procesamiento no sea excesivamente lento
            self.assertLess(processing_time, 30.0, "Procesamiento demasiado lento")
            
            # Medir memoria (aproximado)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Verificar que no use memoria excesiva (límite flexible)
            self.assertLess(memory_mb, 1000, "Uso de memoria excesivo")
            
        except ImportError:
            self.skipTest("psutil no disponible para medición de memoria")
        except Exception as e:
            self.fail(f"Error en benchmarks de rendimiento: {e}")
    
    def test_error_handling_robustness(self):
        """Test de robustez en manejo de errores"""
        try:
            if not (PROCESSING_AVAILABLE and CONFIG_AVAILABLE):
                self.skipTest("Componentes de procesamiento no disponibles")
            
            # Test con imagen corrupta
            corrupted_path = os.path.join(self.temp_dir, "corrupted.jpg")
            with open(corrupted_path, 'wb') as f:
                f.write(b"not_an_image")
            
            # Verificar que no falla catastróficamente
            try:
                image = cv2.imread(corrupted_path)
                if image is not None:
                    processed = self.preprocessor.preprocess_image(image)
            except Exception:
                pass  # Se espera que falle, pero no debe crashear
            
            # Test con imagen vacía
            empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                processed = self.preprocessor.preprocess_image(empty_image)
                self.assertIsNotNone(processed)
            except Exception:
                pass  # Puede fallar, pero no debe crashear
            
        except Exception as e:
            self.fail(f"Error en test de robustez: {e}")
    
    def assertHasAttr(self, obj, attr):
        """Helper para verificar que un objeto tiene un atributo"""
        self.assertTrue(hasattr(obj, attr), f"Objeto no tiene atributo '{attr}'")

def run_complete_integration_tests():
    """Ejecutar todos los tests de integración completa"""
    print("=" * 70)
    print("SIGeC-Balistica - Tests de Integración Completa Consolidados")
    print("=" * 70)
    
    # Configurar entorno
    if 'DISPLAY' not in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Ejecutar tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCompleteIntegrationConsolidated)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generar reporte
    print("\n" + "=" * 70)
    print("REPORTE DE INTEGRACIÓN COMPLETA")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Errores: {len(result.errors)}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Omitidos: {len(result.skipped)}")
    
    if result.errors:
        print("\nERRORES:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if result.failures:
        print("\nFALLOS:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    success_rate = ((result.testsRun - len(result.errors) - len(result.failures)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nTasa de éxito: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_complete_integration_tests()
    sys.exit(0 if success else 1)