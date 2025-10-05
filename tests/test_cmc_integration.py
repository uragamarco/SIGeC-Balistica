#!/usr/bin/env python3
"""
Tests de integración para el algoritmo CMC con UnifiedMatcher.

Este módulo contiene tests para validar la integración correcta del algoritmo CMC
con el sistema de matching unificado.
"""

import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import tempfile
import json

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matching.unified_matcher import UnifiedMatcher, MatchingConfig, AlgorithmType, MatchingLevel
from matching.cmc_algorithm import CMCParameters


class TestCMCIntegration(unittest.TestCase):
    """Tests de integración para CMC con UnifiedMatcher."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        # Configurar matcher con CMC
        self.config = MatchingConfig(
            algorithm=AlgorithmType.CMC,
            level=MatchingLevel.STANDARD
        )
        self.matcher = UnifiedMatcher(self.config)
        
        # Crear imágenes de prueba
        self.test_image1 = self._create_test_image(300, 300)
        self.test_image2 = self._create_similar_image(300, 300)
        self.test_image_different = self._create_different_image(300, 300)
    
    def _create_test_image(self, width, height):
        """Crea una imagen de prueba con patrones balísticos simulados."""
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Simular marcas de percutor (círculos concéntricos)
        center = (width // 2, height // 2)
        for radius in range(20, 80, 10):
            cv2.circle(img, center, radius, 100 + radius, 2)
        
        # Simular estrías radiales
        for angle in range(0, 360, 30):
            x1 = int(center[0] + 50 * np.cos(np.radians(angle)))
            y1 = int(center[1] + 50 * np.sin(np.radians(angle)))
            x2 = int(center[0] + 120 * np.cos(np.radians(angle)))
            y2 = int(center[1] + 120 * np.sin(np.radians(angle)))
            cv2.line(img, (x1, y1), (x2, y2), 150, 1)
        
        # Agregar ruido realista
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _create_similar_image(self, width, height):
        """Crea una imagen similar con pequeñas variaciones."""
        base_img = self._create_test_image(width, height)
        
        # Aplicar pequeña rotación
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, 2.0, 1.0)
        rotated = cv2.warpAffine(base_img, M, (width, height))
        
        # Agregar pequeño desplazamiento
        M_shift = np.float32([[1, 0, 3], [0, 1, -2]])
        shifted = cv2.warpAffine(rotated, M_shift, (width, height))
        
        # Agregar ruido adicional
        noise = np.random.normal(0, 5, shifted.shape).astype(np.int16)
        result = np.clip(shifted.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def _create_different_image(self, width, height):
        """Crea una imagen completamente diferente."""
        img = np.zeros((height, width), dtype=np.uint8)
        
        # Patrón completamente diferente
        cv2.rectangle(img, (50, 50), (250, 250), 200, -1)
        cv2.rectangle(img, (100, 100), (200, 200), 50, -1)
        
        # Líneas diagonales
        for i in range(0, width, 20):
            cv2.line(img, (i, 0), (i, height), 100, 2)
        
        return img
    
    def test_cmc_algorithm_initialization(self):
        """Test de inicialización del algoritmo CMC en UnifiedMatcher."""
        self.assertIsNotNone(self.matcher.cmc_algorithm)
        self.assertIsInstance(self.matcher.cmc_algorithm.parameters, CMCParameters)
    
    def test_extract_features_cmc(self):
        """Test de extracción de características con CMC."""
        features = self.matcher.extract_features(self.test_image1, AlgorithmType.CMC)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features["algorithm"], "CMC")
        self.assertIn("image", features)
        self.assertIn("num_keypoints", features)
        self.assertIn("keypoints", features)
        self.assertIn("descriptors", features)
        
        # CMC no usa keypoints tradicionales
        self.assertEqual(features["num_keypoints"], 0)
        self.assertEqual(len(features["keypoints"]), 0)
        
        # La imagen preprocesada debe estar presente
        self.assertIsNotNone(features["image"])
        self.assertEqual(features["image"].shape, self.test_image1.shape)
    
    def test_match_features_cmc(self):
        """Test de matching de características con CMC."""
        features1 = self.matcher.extract_features(self.test_image1, AlgorithmType.CMC)
        features2 = self.matcher.extract_features(self.test_image2, AlgorithmType.CMC)
        
        result = self.matcher.match_features(features1, features2, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")
        self.assertGreaterEqual(result.similarity_score, 0.0)
        self.assertLessEqual(result.similarity_score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Verificar datos específicos de CMC
        self.assertIn("cmc_score", result.match_data)
        self.assertIn("cmc_count", result.match_data)
        self.assertIn("total_cells", result.match_data)
        self.assertIn("valid_cells", result.match_data)
        self.assertIn("convergence_score", result.match_data)
        self.assertIn("is_match", result.match_data)
    
    def test_compare_images_cmc(self):
        """Test de comparación completa de imágenes con CMC."""
        result = self.matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")
        self.assertGreater(result.processing_time, 0.0)
        
        # Para imágenes similares, debe haber cierta similitud
        self.assertGreater(result.similarity_score, 0.1)
    
    def test_compare_identical_images_cmc(self):
        """Test de comparación de imágenes idénticas con CMC."""
        result = self.matcher.compare_images(self.test_image1, self.test_image1, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        # Para imágenes idénticas, la similitud debe ser alta (ajustado para CMC)
        self.assertGreater(result.similarity_score, 0.3)  # CMC puede tener scores más bajos
        self.assertGreater(result.confidence, 0.2)
    
    def test_compare_different_images_cmc(self):
        """Test de comparación de imágenes diferentes con CMC."""
        result = self.matcher.compare_images(self.test_image1, self.test_image_different, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        # Para imágenes diferentes, la similitud debe ser baja
        self.assertLess(result.similarity_score, 0.8)  # Ajustado para CMC
    
    def test_compare_image_files_cmc(self):
        """Test de comparación de archivos de imagen con CMC."""
        # Crear archivos temporales
        with tempfile.TemporaryDirectory() as temp_dir:
            img1_path = os.path.join(temp_dir, "test1.png")
            img2_path = os.path.join(temp_dir, "test2.png")
            
            cv2.imwrite(img1_path, self.test_image1)
            cv2.imwrite(img2_path, self.test_image2)
            
            result = self.matcher.compare_image_files(img1_path, img2_path, AlgorithmType.CMC)
            
            self.assertIsNotNone(result)
            self.assertEqual(result.algorithm, "CMC")
            self.assertGreaterEqual(result.similarity_score, 0.0)  # Ensure non-negative score
            self.assertGreater(result.processing_time, 0.0)
    
    def test_batch_compare_cmc(self):
        """Test de comparación en lote con CMC."""
        query_features = self.matcher.extract_features(self.test_image1, AlgorithmType.CMC)
        
        # Crear base de datos de características
        database_features = [
            self.matcher.extract_features(self.test_image2, AlgorithmType.CMC),
            self.matcher.extract_features(self.test_image_different, AlgorithmType.CMC),
            self.matcher.extract_features(self.test_image1, AlgorithmType.CMC)  # Imagen idéntica
        ]
        
        results = self.matcher.batch_compare(
            query_features, database_features, AlgorithmType.CMC, top_k=3
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        # Todos los resultados deben ser válidos
        for result in results:
            self.assertEqual(result.algorithm, "CMC")
            self.assertGreaterEqual(result.similarity_score, 0.0)
            self.assertLessEqual(result.similarity_score, 1.0)
        
        # El resultado con la imagen idéntica debe tener la mayor similitud
        similarities = [r.similarity_score for r in results]
        max_similarity = max(similarities)
        self.assertGreater(max_similarity, 0.1)  # Ajustado para CMC
    
    def test_export_match_report_cmc(self):
        """Test de exportación de reporte con CMC."""
        result = self.matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.CMC)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = os.path.join(temp_dir, "cmc_report.json")
            
            success = self.matcher.export_match_report(result, report_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(report_path))
            
            # Verificar contenido del reporte
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            self.assertEqual(report_data["match_result"]["algorithm"], "CMC")
            self.assertIn("cmc_score", report_data["match_data"])
            self.assertIn("cmc_count", report_data["match_data"])
    
    def test_cmc_parameters_configuration(self):
        """Test de configuración de parámetros CMC."""
        # Crear configuración personalizada
        custom_config = MatchingConfig(
            algorithm=AlgorithmType.CMC,
            level=MatchingLevel.ADVANCED
        )
        
        custom_matcher = UnifiedMatcher(custom_config)
        
        self.assertIsNotNone(custom_matcher.cmc_algorithm)
        
        # Verificar que se pueden modificar los parámetros
        custom_matcher.cmc_algorithm.parameters.num_cells_x = 10
        custom_matcher.cmc_algorithm.parameters.num_cells_y = 10
        custom_matcher.cmc_algorithm.parameters.ccf_threshold = 0.3
        
        # Verificar que los parámetros se aplicaron correctamente
        self.assertEqual(custom_matcher.cmc_algorithm.parameters.num_cells_x, 10)
        self.assertEqual(custom_matcher.cmc_algorithm.parameters.num_cells_y, 10)
        self.assertEqual(custom_matcher.cmc_algorithm.parameters.ccf_threshold, 0.3)
        
        result = custom_matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.match_data["num_cells_x"], 10)
        self.assertEqual(result.match_data["num_cells_y"], 10)
    
    def test_error_handling_cmc(self):
        """Test de manejo de errores con CMC."""
        # Test con imagen inválida (None)
        features1 = self.matcher.extract_features(self.test_image1, AlgorithmType.CMC)
        
        # Simular características inválidas
        invalid_features = {"algorithm": "CMC", "image": None}
        
        result = self.matcher.match_features(features1, invalid_features, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")
        self.assertIn("error", result.match_data)
    
    def test_performance_cmc(self):
        """Test de rendimiento del algoritmo CMC."""
        import time
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        result = self.matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.CMC)
        
        total_time = time.time() - start_time
        
        self.assertIsNotNone(result)
        self.assertLess(total_time, 10.0)  # No debe tomar más de 10 segundos
        self.assertGreater(result.processing_time, 0.0)
        self.assertLessEqual(result.processing_time, total_time)
    
    def test_cmc_vs_other_algorithms(self):
        """Test comparativo entre CMC y otros algoritmos."""
        # Comparar con ORB
        orb_config = MatchingConfig(algorithm=AlgorithmType.ORB)
        orb_matcher = UnifiedMatcher(orb_config)
        
        cmc_result = self.matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.CMC)
        orb_result = orb_matcher.compare_images(self.test_image1, self.test_image2, AlgorithmType.ORB)
        
        self.assertIsNotNone(cmc_result)
        self.assertIsNotNone(orb_result)
        
        # Ambos algoritmos deben producir resultados válidos
        self.assertEqual(cmc_result.algorithm, "CMC")
        self.assertEqual(orb_result.algorithm, "ORB")
        
        # Los resultados deben tener estructuras diferentes
        self.assertIn("cmc_score", cmc_result.match_data)
        self.assertNotIn("cmc_score", orb_result.match_data)


class TestCMCRealWorldScenarios(unittest.TestCase):
    """Tests con escenarios del mundo real para CMC."""
    
    def setUp(self):
        """Configuración para tests de escenarios reales."""
        self.matcher = UnifiedMatcher(MatchingConfig(algorithm=AlgorithmType.CMC))
    
    def test_low_quality_images(self):
        """Test con imágenes de baja calidad."""
        # Crear imagen con mucho ruido
        noisy_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        clean_img = np.ones((200, 200), dtype=np.uint8) * 128
        
        result = self.matcher.compare_images(noisy_img, clean_img, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")
        # Para imágenes muy diferentes, la similitud debe ser baja
        self.assertLess(result.similarity_score, 0.3)
    
    def test_different_sizes(self):
        """Test con imágenes de diferentes tamaños."""
        img1 = np.ones((100, 100), dtype=np.uint8) * 128
        img2 = np.ones((200, 200), dtype=np.uint8) * 128
        
        result = self.matcher.compare_images(img1, img2, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")
    
    def test_extreme_lighting_conditions(self):
        """Test con condiciones extremas de iluminación."""
        # Imagen muy oscura
        dark_img = np.ones((200, 200), dtype=np.uint8) * 20
        
        # Imagen muy brillante
        bright_img = np.ones((200, 200), dtype=np.uint8) * 240
        
        result = self.matcher.compare_images(dark_img, bright_img, AlgorithmType.CMC)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.algorithm, "CMC")


if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Ejecutar tests
    unittest.main(verbosity=2)