#!/usr/bin/env python3
"""
Tests unitarios para el algoritmo CMC (Congruent Matching Cells).

Este módulo contiene tests para validar la funcionalidad del algoritmo CMC
implementado según Song et al. (2014).
"""

import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matching.cmc_algorithm import (
    CMCAlgorithm, CMCParameters, CMCCell, CMCResult, CMCMatchResult
)


class TestCMCParameters(unittest.TestCase):
    """Tests para la clase CMCParameters."""
    
    def test_default_parameters(self):
        """Test de parámetros por defecto."""
        params = CMCParameters()
        
        self.assertEqual(params.num_cells_x, 8)
        self.assertEqual(params.num_cells_y, 8)
        self.assertEqual(params.cell_overlap, 0.1)
        self.assertEqual(params.ccf_threshold, 0.2)
        self.assertEqual(params.theta_threshold, 15.0)
        self.assertEqual(params.x_threshold, 20.0)
        self.assertEqual(params.y_threshold, 20.0)
        self.assertEqual(params.cmc_threshold, 6)
        self.assertTrue(params.use_convergence)
        self.assertTrue(params.bidirectional)
    
    def test_custom_parameters(self):
        """Test de parámetros personalizados."""
        params = CMCParameters(
            num_cells_x=6,
            num_cells_y=6,
            ccf_threshold=0.3,
            cmc_threshold=10
        )
        
        self.assertEqual(params.num_cells_x, 6)
        self.assertEqual(params.num_cells_y, 6)
        self.assertEqual(params.ccf_threshold, 0.3)
        self.assertEqual(params.cmc_threshold, 10)


class TestCMCAlgorithm(unittest.TestCase):
    """Tests para la clase CMCAlgorithm."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        self.params = CMCParameters(num_cells_x=4, num_cells_y=4)
        self.cmc = CMCAlgorithm(self.params)
        
        # Crear imágenes de prueba
        self.test_image1 = self._create_test_image(200, 200)
        self.test_image2 = self._create_test_image_shifted(200, 200, shift_x=10, shift_y=5)
        self.test_image_identical = self.test_image1.copy()
    
    def _create_test_image(self, width, height):
        """Crea una imagen de prueba con patrones."""
        img = np.zeros((height, width), dtype=np.float32)
        
        # Agregar algunos patrones
        cv2.rectangle(img, (50, 50), (100, 100), 1.0, -1)
        cv2.circle(img, (150, 150), 30, 0.5, -1)
        cv2.line(img, (0, 0), (width, height), 0.25, 2)
        
        return img
    
    def _create_test_image_shifted(self, width, height, shift_x=0, shift_y=0):
        """Crea una imagen de prueba desplazada."""
        base_img = self._create_test_image(width, height)
        
        # Crear matriz de transformación para desplazamiento
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_img = cv2.warpAffine(base_img, M, (width, height))
        
        return shifted_img
    
    def test_preprocess_image(self):
        """Test de preprocesamiento de imagen."""
        processed = self.cmc.preprocess_image(self.test_image1)
        
        self.assertEqual(processed.shape, self.test_image1.shape)
        self.assertEqual(processed.dtype, np.float32)
        
        # Verificar que la imagen fue procesada
        self.assertIsInstance(processed, np.ndarray)
    
    def test_divide_into_cells(self):
        """Test de división en celdas."""
        cells = self.cmc.divide_into_cells(self.test_image1)
        
        self.assertIsInstance(cells, list)
        self.assertGreater(len(cells), 0)
        
        # Verificar que todas las celdas son válidas
        for cell in cells:
            self.assertIsInstance(cell, CMCCell)
            self.assertGreaterEqual(cell.row, 0)
            self.assertGreaterEqual(cell.col, 0)
            self.assertIsInstance(cell.data, np.ndarray)
    
    def test_calculate_cross_correlation(self):
        """Test de cálculo de correlación cruzada."""
        cells = self.cmc.divide_into_cells(self.test_image1)
        
        if len(cells) >= 2:
            cell1_data = cells[0].data
            cell2_data = cells[1].data
            
            ccf_max, x_offset, y_offset = self.cmc.calculate_cross_correlation(cell1_data, cell2_data)
            
            self.assertIsInstance(ccf_max, float)
            self.assertIsInstance(x_offset, float)
            self.assertIsInstance(y_offset, float)
            
            # Los valores deben estar en rangos razonables
            self.assertGreaterEqual(ccf_max, -1.0)
            self.assertLessEqual(ccf_max, 1.0)
    
    def test_calculate_rotation_angle(self):
        """Test de estimación de ángulo de rotación."""
        cells = self.cmc.divide_into_cells(self.test_image1)
        
        if len(cells) >= 1:
            cell_data = cells[0].data
            
            # Test con desplazamientos conocidos
            angle = self.cmc.calculate_rotation_angle(cell_data, cell_data, 0.0, 0.0)
            
            self.assertIsInstance(angle, float)
            # Para la misma celda, el ángulo debe ser cercano a 0
            self.assertLess(abs(angle), 10.0)
    
    def test_is_congruent_matching_cell(self):
        """Test de determinación de celdas CMC."""
        # Test con valores que deberían ser congruentes
        is_cmc = self.cmc.is_congruent_matching_cell(
            ccf_max=0.8,
            theta=2.0,
            x_offset=5.0,
            y_offset=3.0,
            median_theta=1.0,
            median_x=4.0,
            median_y=2.0
        )
        
        self.assertIsInstance(is_cmc, bool)
        self.assertTrue(is_cmc)  # Valores cercanos a las medianas deben ser congruentes
        
        # Test con valores que no deberían ser congruentes
        is_not_cmc = self.cmc.is_congruent_matching_cell(
            ccf_max=0.1,  # Correlación muy baja
            theta=50.0,   # Ángulo muy diferente
            x_offset=100.0,  # Desplazamiento muy grande
            y_offset=100.0,
            median_theta=1.0,
            median_x=4.0,
            median_y=2.0
        )
        
        self.assertFalse(is_not_cmc)
    
    def test_calculate_convergence_score(self):
        """Test de cálculo de score de convergencia."""
        # Crear resultados de prueba
        test_results = [
            CMCResult(
                cell_index=(0, 0),
                ccf_max=0.8,
                theta=2.0,
                x_offset=5.0,
                y_offset=3.0,
                is_cmc=True,
                confidence=0.9
            ),
            CMCResult(
                cell_index=(0, 1),
                ccf_max=0.7,
                theta=1.5,
                x_offset=4.5,
                y_offset=2.8,
                is_cmc=True,
                confidence=0.8
            )
        ]
        
        convergence = self.cmc.calculate_convergence_score(test_results)
        
        self.assertIsInstance(convergence, float)
        self.assertGreaterEqual(convergence, 0.0)
        self.assertLessEqual(convergence, 1.0)
    
    def test_compare_images(self):
        """Test de comparación de imágenes."""
        result = self.cmc.compare_images(self.test_image1, self.test_image_identical)
        
        self.assertIsInstance(result, CMCMatchResult)
        self.assertGreaterEqual(result.cmc_count, 0)
        self.assertGreater(result.total_cells, 0)
        self.assertGreaterEqual(result.cmc_score, 0.0)
        self.assertLessEqual(result.cmc_score, 1.0)
        self.assertIsInstance(result.is_match, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Para imágenes idénticas, debe haber alta similitud
        self.assertGreater(result.cmc_score, 0.5)
    
    def test_bidirectional_comparison(self):
        """Test de comparación bidireccional."""
        result = self.cmc.bidirectional_comparison(self.test_image1, self.test_image_identical)
        
        self.assertIsInstance(result, CMCMatchResult)
        self.assertIsNotNone(result.convergence_score)
        self.assertGreaterEqual(result.convergence_score, 0.0)
        self.assertLessEqual(result.convergence_score, 1.0)
        
        # Para imágenes idénticas, el score debe ser alto
        self.assertGreater(result.cmc_score, 0.5)
    
    def test_different_images(self):
        """Test con imágenes completamente diferentes."""
        # Crear imagen completamente diferente
        different_img = np.random.rand(200, 200).astype(np.float32)
        
        result = self.cmc.compare_images(self.test_image1, different_img)
        
        self.assertIsInstance(result, CMCMatchResult)
        # Para imágenes diferentes, el score debe ser bajo
        self.assertLess(result.cmc_score, 0.5)
    
    def test_edge_cases(self):
        """Test de casos extremos."""
        # Imagen muy pequeña
        small_img = np.ones((50, 50), dtype=np.float32) * 0.5
        
        result = self.cmc.compare_images(small_img, small_img)
        self.assertIsInstance(result, CMCMatchResult)
        
        # Imagen completamente negra
        black_img = np.zeros((100, 100), dtype=np.float32)
        
        result = self.cmc.compare_images(black_img, black_img)
        self.assertIsInstance(result, CMCMatchResult)
        
        # Imagen completamente blanca
        white_img = np.ones((100, 100), dtype=np.float32)
        
        result = self.cmc.compare_images(white_img, white_img)
        self.assertIsInstance(result, CMCMatchResult)


class TestCMCIntegration(unittest.TestCase):
    """Tests de integración para el algoritmo CMC."""
    
    def setUp(self):
        """Configuración inicial para tests de integración."""
        self.cmc = CMCAlgorithm()
    
    def test_performance_large_images(self):
        """Test de rendimiento con imágenes grandes."""
        # Crear imágenes grandes
        large_img1 = np.random.rand(400, 400).astype(np.float32)
        large_img2 = large_img1.copy()
        
        import time
        start_time = time.time()
        
        result = self.cmc.compare_images(large_img1, large_img2)
        
        processing_time = time.time() - start_time
        
        self.assertIsInstance(result, CMCMatchResult)
        # El procesamiento no debe tomar más de 30 segundos
        self.assertLess(processing_time, 30.0)
    
    def test_memory_usage(self):
        """Test de uso de memoria."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Procesar múltiples imágenes
        for i in range(10):
            img1 = np.random.rand(200, 200).astype(np.float32)
            img2 = np.random.rand(200, 200).astype(np.float32)
            
            result = self.cmc.compare_images(img1, img2)
            self.assertIsInstance(result, CMCMatchResult)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # El aumento de memoria no debe ser excesivo (menos de 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Ejecutar tests
    unittest.main(verbosity=2)