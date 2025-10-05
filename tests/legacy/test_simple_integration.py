#!/usr/bin/env python3
"""
Test de Integración Simplificado
Sistema Balístico Forense MVP

Verifica las correcciones aplicadas a los componentes principales
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import cv2
from pathlib import Path

# Configurar Qt para modo headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar módulos del proyecto
from utils.config import get_config
from image_processing.unified_preprocessor import UnifiedPreprocessor
from image_processing.feature_extractor import FeatureExtractor
from matching.unified_matcher import UnifiedMatcher
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector

class TestSimpleIntegration(unittest.TestCase):
    """Test simplificado de integración"""
    
    def setUp(self):
        """Configuración inicial"""
        self.config = get_config()
        
        # Inicializar componentes
        self.preprocessor = UnifiedPreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.matcher = UnifiedMatcher(self.config)
        self.db_manager = VectorDatabase(self.config)
        
        # Crear imagen de prueba
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self):
        """Crea una imagen de prueba sintética y la guarda en un archivo temporal"""
        # Crear imagen sintética de 400x400 píxeles
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Agregar algunos patrones para simular características balísticas
        # Círculo central (simula culote de vaina)
        cv2.circle(image, (200, 200), 80, (150, 150, 150), -1)
        
        # Líneas radiales (simulan estrías)
        for angle in range(0, 360, 30):
            x1 = int(200 + 60 * np.cos(np.radians(angle)))
            y1 = int(200 + 60 * np.sin(np.radians(angle)))
            x2 = int(200 + 120 * np.cos(np.radians(angle)))
            y2 = int(200 + 120 * np.sin(np.radians(angle)))
            cv2.line(image, (x1, y1), (x2, y2), (100, 100, 100), 2)
        
        # Agregar algo de ruido
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Guardar en archivo temporal
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)  # Cerrar el descriptor de archivo
        
        # Convertir de RGB a BGR para OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(temp_path, image_bgr)
        
        return temp_path
    
    def tearDown(self):
        """Limpieza"""
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    def test_config_loading(self):
        """Test de carga de configuración"""
        print("Testing config loading...")
        self.assertIsNotNone(self.config)
        self.assertIsNotNone(self.config.database)
        self.assertIsNotNone(self.config.image_processing)
        self.assertIsNotNone(self.config.matching)
        print("  ✅ Config loaded successfully")
    
    def test_database_initialization(self):
        """Test de inicialización de base de datos"""
        print("Testing database initialization...")
        self.assertIsNotNone(self.db_manager)
        print("  ✅ Database initialized successfully")
    
    def test_image_preprocessing(self):
        """Test básico de preprocesamiento de imagen"""
        print("\n--- Test: Preprocesamiento de imagen ---")
        
        # Crear imagen de prueba
        test_image_path = self._create_test_image()
        
        try:
            # Preprocesar imagen usando el preprocessor ya inicializado
            result = self.preprocessor.preprocess_ballistic_image(test_image_path, evidence_type='vaina')
            
            # Verificar resultado
            self.assertTrue(result.success, f"Preprocesamiento falló: {result.error_message}")
            self.assertIsNotNone(result.processed_image, "Imagen procesada es None")
            self.assertIsNotNone(result.original_image, "Imagen original es None")
            
            print(f"✓ Preprocesamiento exitoso")
            print(f"  - Imagen original: {result.original_image.shape}")
            print(f"  - Imagen procesada: {result.processed_image.shape}")
            print(f"  - Tiempo: {result.processing_time:.3f}s")
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
    
    def test_feature_extraction(self):
        """Test básico de extracción de características"""
        print("\n--- Test: Extracción de características ---")
        
        # Crear imagen de prueba
        test_image_path = self._create_test_image()
        
        try:
            # Preprocesar imagen primero
            preprocessor = UnifiedPreprocessor(self.config)
            result = preprocessor.preprocess_ballistic_image(test_image_path, evidence_type='vaina')
            self.assertTrue(result.success, "Preprocesamiento debe ser exitoso")
            
            # Extraer características
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(result.processed_image)
            
            # Verificar resultado
            self.assertIsNotNone(keypoints, "Keypoints no deben ser None")
            self.assertGreater(len(keypoints), 0, "Debe haber al menos un keypoint")
            self.assertIsNotNone(descriptors, "Descriptores no deben ser None")
            
            print(f"✓ Extracción exitosa")
            print(f"  - Keypoints: {len(keypoints)}")
            print(f"  - Descriptores: {descriptors.shape}")
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
    
    def test_matching(self):
        """Test básico de matching entre dos imágenes"""
        print("\n--- Test: Matching de imágenes ---")
        
        # Crear dos imágenes de prueba
        test_image_path1 = self._create_test_image()
        test_image_path2 = self._create_test_image()  # Misma imagen para garantizar similitud
        
        try:
            # Preprocesar ambas imágenes
            preprocessor = UnifiedPreprocessor(self.config)
            result1 = preprocessor.preprocess_ballistic_image(test_image_path1, evidence_type='vaina')
            result2 = preprocessor.preprocess_ballistic_image(test_image_path2, evidence_type='vaina')
            
            self.assertTrue(result1.success, "Preprocesamiento 1 debe ser exitoso")
            self.assertTrue(result2.success, "Preprocesamiento 2 debe ser exitoso")
            
            # Extraer características
            keypoints1, descriptors1 = self.feature_extractor.get_keypoints_and_descriptors(result1.processed_image)
            keypoints2, descriptors2 = self.feature_extractor.get_keypoints_and_descriptors(result2.processed_image)
            
            # Realizar matching usando BFMatcher directamente
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calcular similitud básica
            similarity_score = len(matches) / max(len(keypoints1), len(keypoints2)) if keypoints1 and keypoints2 else 0.0
            
            # Verificar resultado
            self.assertGreaterEqual(similarity_score, 0.1, 
                                  f"Similitud debe ser >= 0.1, obtenido: {similarity_score}")
            
            print(f"✓ Matching exitoso")
            print(f"  - Similitud: {similarity_score:.3f}")
            print(f"  - Matches válidos: {len(matches)}")
            
        finally:
            # Limpiar archivos temporales
            for path in [test_image_path1, test_image_path2]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_database_operations(self):
        """Test básico de operaciones de base de datos"""
        print("\n--- Test: Operaciones de base de datos ---")
        
        # Crear imagen de prueba
        test_image_path = self._create_test_image()
        
        try:
            # Preprocesar imagen usando el preprocessor ya inicializado
            result = self.preprocessor.preprocess_ballistic_image(test_image_path, evidence_type='vaina')
            self.assertTrue(result.success, "Preprocesamiento debe ser exitoso")
            
            # Extraer características
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(result.processed_image)
            
            # Agregar caso a la base de datos con número único
            import time
            unique_case_number = f"TEST_{int(time.time())}"
            case = BallisticCase(
                case_number=unique_case_number,
                investigator="Test User",
                description="Test case for integration"
            )
            case_id = self.db_manager.add_case(case)
            
            # Crear objeto imagen con hash único
            import hashlib
            unique_hash = hashlib.md5(f"{test_image_path}_{time.time()}".encode()).hexdigest()
            image = BallisticImage(
                case_id=case_id,
                filename=os.path.basename(test_image_path),
                file_path=test_image_path,
                evidence_type="vaina",
                image_hash=unique_hash
            )
            image_id = self.db_manager.add_image(image)
            
            # Crear vector de características
            feature_vector = FeatureVector(
                image_id=image_id,
                algorithm="ORB"
            )
            vector_id = self.db_manager.add_feature_vector(feature_vector, descriptors)
            
            # Buscar imágenes similares
            similar_images = self.db_manager.search_similar_vectors(descriptors, k=5)
            
            # Verificar resultado
            self.assertIsNotNone(case_id, "ID de caso no debe ser None")
            self.assertIsNotNone(image_id, "ID de imagen no debe ser None")
            self.assertIsNotNone(vector_id, "ID de vector no debe ser None")
            self.assertIsInstance(similar_images, list, "Resultado debe ser una lista")
            
            print(f"✓ Operaciones de BD exitosas")
            print(f"  - Caso ID: {case_id}")
            print(f"  - Imagen ID: {image_id}")
            print(f"  - Vector ID: {vector_id}")
            print(f"  - Imágenes similares encontradas: {len(similar_images)}")
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

if __name__ == '__main__':
    print("=" * 60)
    print("SIGeC-Balistica - Test Simplificado de Integración")
    print("=" * 60)
    
    # Configurar entorno para GUI headless
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    unittest.main(verbosity=2)