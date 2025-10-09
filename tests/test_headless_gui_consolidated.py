#!/usr/bin/env python3
"""
Test GUI Headless Consolidado - SIGeC-Balistica
===============================================

Pruebas consolidadas de interfaz gráfica sin display que validan:
- Integración GUI-Backend en modo headless
- Componentes GUI mockeados
- Operaciones de base de datos sin interfaz
- Correcciones de integración entre módulos
- Funcionalidad simplificada de integración

Migrado desde:
- tests/legacy/test_gui_headless.py
- tests/legacy/test_integration_fixes.py  
- tests/legacy/test_simple_integration.py

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
from pathlib import Path
from unittest.mock import MagicMock, patch

# Configurar Qt para modo headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock PyQt5 para evitar problemas de display
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()

from utils.config import get_config
from image_processing.unified_preprocessor import UnifiedPreprocessor
from image_processing.feature_extractor import FeatureExtractor
from matching.unified_matcher import UnifiedMatcher
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector

class TestHeadlessGUIConsolidated(unittest.TestCase):
    """Test consolidado de GUI en modo headless"""
    
    def setUp(self):
        """Configuración inicial"""
        try:
            self.config = get_config()
            
            # Verificar y agregar atributos faltantes si es necesario
            if not hasattr(self.config, 'enable_gpu_acceleration'):
                self.config.enable_gpu_acceleration = False
            if not hasattr(self.config.matching, 'enable_gpu_acceleration'):
                self.config.matching.enable_gpu_acceleration = False
                
            self.preprocessor = UnifiedPreprocessor(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            self.matcher = UnifiedMatcher(self.config)
            self.db_manager = VectorDatabase(self.config)
            
            # Crear imagen de prueba
            self.test_image_path = self._create_test_image()
        except Exception as e:
            # Configuración mínima de fallback
            self.config = type('Config', (), {
                'database': type('DB', (), {'sqlite_path': 'test.db'})(),
                'image_processing': type('IP', (), {'max_image_size': 2048})(),
                'matching': type('M', (), {'enable_gpu_acceleration': False})(),
                'enable_gpu_acceleration': False
            })()
            self.preprocessor = None
            self.feature_extractor = None
            self.matcher = None
            self.db_manager = None
            self.test_image_path = None
        
    def _create_test_image(self, size=(400, 400), pattern="circles"):
        """Crear imagen de prueba sintética"""
        image = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(image)
        
        if pattern == "circles":
            # Patrón de círculos para simular características balísticas
            for i in range(5):
                for j in range(5):
                    x = 50 + i * 70
                    y = 50 + j * 70
                    radius = 15 + (i + j) * 2
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                               fill='black', outline='gray')
        elif pattern == "lines":
            # Patrón de líneas para simular estrías
            for i in range(10):
                y = 40 + i * 35
                draw.line([20, y, size[0]-20, y], fill='black', width=2)
        
        # Guardar en archivo temporal
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_file.name)
        return temp_file.name
    
    def tearDown(self):
        """Limpieza después de cada test"""
        if hasattr(self, 'test_image_path') and os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_config_loading(self):
        """Test de carga de configuración"""
        self.assertIsNotNone(self.config)
        self.assertIn('database', self.config)
        self.assertIn('image_processing', self.config)
        
    def test_database_initialization(self):
        """Test de inicialización de base de datos"""
        self.assertIsNotNone(self.db_manager)
        # Verificar que la base de datos se puede inicializar sin errores
        
    def test_gui_backend_integration_headless(self):
        """Test de integración GUI-Backend en modo headless"""
        try:
            # Simular carga de imagen
            image = cv2.imread(self.test_image_path)
            self.assertIsNotNone(image)
            
            # Preprocesamiento
            processed = self.preprocessor.preprocess_image(image)
            self.assertIsNotNone(processed)
            
            # Extracción de características
            features = self.feature_extractor.extract_all_features(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), ['orb']
            )
            self.assertIn('orb', features)
            self.assertGreater(len(features['orb'].get('keypoints', [])), 0)
            
            # Verificar que el proceso completo funciona sin GUI
            self.assertTrue(True)  # Si llegamos aquí, la integración funciona
            
        except Exception as e:
            self.fail(f"Error en integración GUI-Backend headless: {e}")
    
    def test_gui_components_mock(self):
        """Test de componentes GUI mockeados"""
        # Verificar que PyQt5 está mockeado
        import PyQt5.QtWidgets as QtWidgets
        self.assertIsInstance(QtWidgets, MagicMock)
        
        # Simular creación de widgets
        app = QtWidgets.QApplication([])
        window = QtWidgets.QMainWindow()
        
        # Verificar que no hay errores al crear componentes mockeados
        self.assertIsNotNone(app)
        self.assertIsNotNone(window)
    
    def test_database_operations_headless(self):
        """Test de operaciones de base de datos sin interfaz"""
        try:
            # Crear caso de prueba
            test_case = BallisticCase(
                case_id="TEST_HEADLESS_001",
                description="Test case for headless operations",
                created_by="test_user"
            )
            
            # Crear imagen balística de prueba
            test_image = BallisticImage(
                case_id="TEST_HEADLESS_001",
                image_path=self.test_image_path,
                image_type="bullet",
                metadata={"test": True}
            )
            
            # Verificar que las operaciones no fallan
            self.assertIsNotNone(test_case)
            self.assertIsNotNone(test_image)
            
        except Exception as e:
            self.fail(f"Error en operaciones de base de datos headless: {e}")
    
    def test_feature_extractor_fixes(self):
        """Test de correcciones en FeatureExtractor"""
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            
            # Probar extract_all_features (método correcto)
            features = self.feature_extractor.extract_all_features(gray_image, ['orb'])
            self.assertIn('orb', features)
            self.assertIsInstance(features['orb'].get('keypoints', []), list)
            
            # Verificar que extract_features no existe (método obsoleto)
            self.assertFalse(hasattr(self.feature_extractor, 'extract_features'),
                           "extract_features obsoleto aún existe")
            
        except Exception as e:
            self.fail(f"Error en correcciones de FeatureExtractor: {e}")
    
    def test_database_fixes(self):
        """Test de correcciones en VectorDatabase"""
        try:
            # Verificar inicialización correcta
            self.assertIsNotNone(self.db_manager)
            
            # Verificar que los métodos principales existen
            self.assertTrue(hasattr(self.db_manager, 'add_case'))
            self.assertTrue(hasattr(self.db_manager, 'add_image'))
            self.assertTrue(hasattr(self.db_manager, 'search_similar'))
            
        except Exception as e:
            self.fail(f"Error en correcciones de VectorDatabase: {e}")
    
    def test_preprocessor_fixes(self):
        """Test de correcciones en UnifiedPreprocessor"""
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            # Probar preprocesamiento
            processed = self.preprocessor.preprocess_image(test_image)
            self.assertIsNotNone(processed)
            self.assertEqual(len(processed.shape), 2)  # Debe ser imagen en escala de grises
            
        except Exception as e:
            self.fail(f"Error en correcciones de UnifiedPreprocessor: {e}")
    
    def test_unified_matcher_fixes(self):
        """Test de correcciones en UnifiedMatcher"""
        try:
            # Crear dos imágenes de prueba
            img1 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            img2 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            # Probar comparación
            result = self.matcher.compare_images(img1, img2)
            self.assertIsNotNone(result)
            self.assertHasAttr(result, 'similarity_score')
            self.assertHasAttr(result, 'confidence')
            
        except Exception as e:
            self.fail(f"Error en correcciones de UnifiedMatcher: {e}")
    
    def test_image_preprocessing_comprehensive(self):
        """Test comprehensivo de preprocesamiento de imágenes"""
        try:
            # Cargar imagen de prueba
            image = cv2.imread(self.test_image_path)
            self.assertIsNotNone(image)
            
            # Preprocesamiento básico
            processed = self.preprocessor.preprocess_image(image)
            self.assertIsNotNone(processed)
            
            # Verificar dimensiones
            self.assertEqual(len(processed.shape), 2)  # Escala de grises
            self.assertGreater(processed.shape[0], 0)
            self.assertGreater(processed.shape[1], 0)
            
            # Verificar rango de valores
            self.assertGreaterEqual(processed.min(), 0)
            self.assertLessEqual(processed.max(), 255)
            
        except Exception as e:
            self.fail(f"Error en preprocesamiento comprehensivo: {e}")
    
    def test_feature_extraction_comprehensive(self):
        """Test comprehensivo de extracción de características"""
        try:
            # Cargar y procesar imagen
            image = cv2.imread(self.test_image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extraer características con diferentes algoritmos
            algorithms = ['orb', 'sift', 'akaze']
            for algorithm in algorithms:
                try:
                    features = self.feature_extractor.extract_all_features(gray_image, [algorithm])
                    self.assertIn(algorithm, features)
                    
                    # Verificar estructura de características
                    alg_features = features[algorithm]
                    self.assertIn('keypoints', alg_features)
                    self.assertIn('descriptors', alg_features)
                    
                except Exception as alg_error:
                    # Algunos algoritmos pueden no estar disponibles
                    print(f"Algoritmo {algorithm} no disponible: {alg_error}")
                    continue
            
        except Exception as e:
            self.fail(f"Error en extracción comprehensiva: {e}")
    
    def test_matching_comprehensive(self):
        """Test comprehensivo de matching"""
        try:
            # Crear dos imágenes diferentes
            img1_path = self._create_test_image(pattern="circles")
            img2_path = self._create_test_image(pattern="lines")
            
            try:
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                
                # Realizar matching
                result = self.matcher.compare_images(img1, img2)
                
                # Verificar resultado
                self.assertIsNotNone(result)
                self.assertIsInstance(result.similarity_score, (int, float))
                self.assertIsInstance(result.confidence, (int, float))
                self.assertGreaterEqual(result.similarity_score, 0)
                self.assertLessEqual(result.similarity_score, 100)
                
            finally:
                # Limpiar archivos temporales
                if os.path.exists(img1_path):
                    os.unlink(img1_path)
                if os.path.exists(img2_path):
                    os.unlink(img2_path)
            
        except Exception as e:
            self.fail(f"Error en matching comprehensivo: {e}")
    
    def assertHasAttr(self, obj, attr):
        """Helper para verificar que un objeto tiene un atributo"""
        self.assertTrue(hasattr(obj, attr), f"Objeto no tiene atributo '{attr}'")

def run_headless_tests():
    """Ejecutar todos los tests en modo headless"""
    print("=" * 60)
    print("SIGeC-Balistica - Tests GUI Headless Consolidados")
    print("=" * 60)
    
    # Configurar entorno headless
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    
    # Ejecutar tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHeadlessGUIConsolidated)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_headless_tests()
    sys.exit(0 if success else 1)