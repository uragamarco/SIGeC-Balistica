#!/usr/bin/env python3
"""
Tests comprehensivos para el Pipeline Científico Unificado
========================================================

Este módulo contiene tests unitarios y de integración para validar
la funcionalidad completa del pipeline científico de análisis balístico.

Incluye tests para:
- Configuración del pipeline
- Inicialización de componentes
- Procesamiento de imágenes
- Evaluación de calidad NIST
- Detección de ROI
- Matching de características
- Análisis CMC
- Determinación de conclusiones AFTE
- Exportación de reportes
"""

import unittest
import tempfile
import os
import json
import numpy as np
import cv2
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos del pipeline
try:
    from core.unified_pipeline import (
        ScientificPipeline, PipelineConfig, PipelineResult, 
        PipelineLevel, AFTEConclusion
    )
    from core.pipeline_config import (
        PipelineConfiguration, create_pipeline_config,
        get_available_levels, get_predefined_config
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Error importando pipeline: {e}")
    PIPELINE_AVAILABLE = False

# Importar componentes individuales
try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingLevel
    from image_processing.unified_roi_detector import UnifiedROIDetector, DetectionLevel
    from matching.unified_matcher import UnifiedMatcher, AlgorithmType, MatchingConfig
    from matching.cmc_algorithm import CMCAlgorithm, CMCParameters
    from nist_standards.quality_metrics import NISTQualityMetrics
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Error importando componentes: {e}")
    COMPONENTS_AVAILABLE = False


class TestPipelineConfiguration(unittest.TestCase):
    """Tests para la configuración del pipeline"""
    
    def setUp(self):
        """Configuración inicial para los tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_create_pipeline_config_basic(self):
        """Test de creación de configuración básica"""
        config = create_pipeline_config("basic")
        
        self.assertEqual(config.level.value, "basic")
        self.assertIsNotNone(config.quality_assessment)
        self.assertIsNotNone(config.preprocessing)
        self.assertIsNotNone(config.roi_detection)
        self.assertIsNotNone(config.matching)
        self.assertIsNotNone(config.cmc_analysis)
        self.assertIsNotNone(config.afte_conclusion)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_create_pipeline_config_forensic(self):
        """Test de creación de configuración forense"""
        config = create_pipeline_config("forensic")
        
        self.assertEqual(config.level.value, "forensic")
        # Verificar que los umbrales forenses son más estrictos
        self.assertGreaterEqual(config.quality_assessment.min_snr, 30.0)
        self.assertGreaterEqual(config.afte_conclusion.identification_threshold, 0.9)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_available_levels(self):
        """Test de niveles disponibles"""
        levels = get_available_levels()
        
        self.assertIn("basic", levels)
        self.assertIn("standard", levels)
        self.assertIn("advanced", levels)
        self.assertIn("forensic", levels)
        self.assertEqual(len(levels), 4)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_predefined_configs(self):
        """Test de configuraciones predefinidas"""
        config = get_predefined_config("forensic_analysis")
        
        self.assertEqual(config.level.value, "forensic")
        self.assertTrue(config.quality_assessment.strict_mode)
        self.assertTrue(config.export_intermediate_results)
        self.assertTrue(config.export_detailed_report)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_config_serialization(self):
        """Test de serialización de configuración"""
        config = create_pipeline_config("standard")
        
        # Convertir a diccionario
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["level"], "standard")
        
        # Recrear desde diccionario
        config_restored = PipelineConfiguration.from_dict(config_dict)
        self.assertEqual(config_restored.level.value, "standard")


class TestScientificPipeline(unittest.TestCase):
    """Tests para el pipeline científico principal"""
    
    def setUp(self):
        """Configuración inicial para los tests"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear imágenes de prueba
        self.test_image1 = self._create_test_image(300, 300, pattern="circles")
        self.test_image2 = self._create_test_image(300, 300, pattern="similar_circles")
        self.test_image_different = self._create_test_image(300, 300, pattern="squares")
        
        # Guardar imágenes
        self.image1_path = os.path.join(self.temp_dir, "test1.jpg")
        self.image2_path = os.path.join(self.temp_dir, "test2.jpg")
        self.image_diff_path = os.path.join(self.temp_dir, "test_diff.jpg")
        
        cv2.imwrite(self.image1_path, self.test_image1)
        cv2.imwrite(self.image2_path, self.test_image2)
        cv2.imwrite(self.image_diff_path, self.test_image_different)
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, width, height, pattern="circles"):
        """Crear imagen de prueba con patrones específicos"""
        image = np.zeros((height, width), dtype=np.uint8)
        
        if pattern == "circles":
            # Crear círculos concéntricos
            center = (width // 2, height // 2)
            for radius in range(20, min(width, height) // 2, 30):
                cv2.circle(image, center, radius, 255, 2)
        elif pattern == "similar_circles":
            # Crear círculos similares pero ligeramente desplazados
            center = (width // 2 + 5, height // 2 + 5)
            for radius in range(20, min(width, height) // 2, 30):
                cv2.circle(image, center, radius, 255, 2)
        elif pattern == "squares":
            # Crear cuadrados
            for size in range(20, min(width, height) // 2, 40):
                top_left = (width // 2 - size // 2, height // 2 - size // 2)
                bottom_right = (width // 2 + size // 2, height // 2 + size // 2)
                cv2.rectangle(image, top_left, bottom_right, 255, 2)
        
        # Añadir algo de ruido
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_pipeline_initialization(self):
        """Test de inicialización del pipeline"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        self.assertIsNotNone(pipeline.config)
        self.assertEqual(pipeline.config.level.value, "standard")
        
        # Verificar que los componentes se inicializan correctamente
        self.assertIsNotNone(pipeline.quality_metrics)
        self.assertIsNotNone(pipeline.preprocessor)
        self.assertIsNotNone(pipeline.roi_detector)
        self.assertIsNotNone(pipeline.matcher)
        self.assertIsNotNone(pipeline.cmc_algorithm)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_load_images(self):
        """Test de carga de imágenes"""
        config = create_pipeline_config("basic")
        pipeline = ScientificPipeline(config)
        
        # Cargar imágenes válidas
        result = pipeline.load_images(self.image1_path, self.image2_path)
        self.assertTrue(result)
        self.assertIsNotNone(pipeline.image1)
        self.assertIsNotNone(pipeline.image2)
        
        # Verificar dimensiones
        self.assertEqual(len(pipeline.image1.shape), 2)  # Escala de grises
        self.assertEqual(len(pipeline.image2.shape), 2)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_load_invalid_images(self):
        """Test de carga de imágenes inválidas"""
        config = create_pipeline_config("basic")
        pipeline = ScientificPipeline(config)
        
        # Intentar cargar archivo inexistente
        result = pipeline.load_images("nonexistent.jpg", self.image2_path)
        self.assertFalse(result)
        
        # Intentar cargar archivo no imagen
        text_file = os.path.join(self.temp_dir, "test.txt")
        with open(text_file, 'w') as f:
            f.write("not an image")
        
        result = pipeline.load_images(text_file, self.image2_path)
        self.assertFalse(result)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_quality_assessment(self):
        """Test de evaluación de calidad"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Cargar imágenes
        pipeline.load_images(self.image1_path, self.image2_path)
        
        # Evaluar calidad
        quality1, quality2 = pipeline.assess_quality()
        
        self.assertIsNotNone(quality1)
        self.assertIsNotNone(quality2)
        self.assertGreaterEqual(quality1.overall_quality, 0.0)
        self.assertLessEqual(quality1.overall_quality, 1.0)
        self.assertGreaterEqual(quality2.overall_quality, 0.0)
        self.assertLessEqual(quality2.overall_quality, 1.0)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_preprocessing(self):
        """Test de preprocesamiento"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Cargar imágenes
        pipeline.load_images(self.image1_path, self.image2_path)
        
        # Preprocesar
        processed1, processed2 = pipeline.preprocess_images()
        
        self.assertIsNotNone(processed1)
        self.assertIsNotNone(processed2)
        self.assertEqual(processed1.shape, pipeline.image1.shape)
        self.assertEqual(processed2.shape, pipeline.image2.shape)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_roi_detection(self):
        """Test de detección de ROI"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Cargar y preprocesar imágenes
        pipeline.load_images(self.image1_path, self.image2_path)
        processed1, processed2 = pipeline.preprocess_images()
        
        # Detectar ROI
        roi1, roi2 = pipeline.detect_roi(processed1, processed2)
        
        self.assertIsInstance(roi1, list)
        self.assertIsInstance(roi2, list)
        # Debe detectar al menos una región
        self.assertGreaterEqual(len(roi1), 0)
        self.assertGreaterEqual(len(roi2), 0)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_feature_matching(self):
        """Test de matching de características"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Cargar y preprocesar imágenes
        pipeline.load_images(self.image1_path, self.image2_path)
        processed1, processed2 = pipeline.preprocess_images()
        
        # Realizar matching
        match_result = pipeline.match_features(processed1, processed2)
        
        self.assertIsNotNone(match_result)
        self.assertGreaterEqual(match_result.similarity_score, 0.0)
        self.assertLessEqual(match_result.similarity_score, 1.0)
        self.assertGreater(match_result.processing_time, 0.0)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_cmc_analysis(self):
        """Test de análisis CMC"""
        config = create_pipeline_config("advanced")
        pipeline = ScientificPipeline(config)
        
        # Cargar y preprocesar imágenes
        pipeline.load_images(self.image1_path, self.image2_path)
        processed1, processed2 = pipeline.preprocess_images()
        
        # Realizar análisis CMC
        cmc_result = pipeline.perform_cmc_analysis(processed1, processed2)
        
        self.assertIsNotNone(cmc_result)
        self.assertGreaterEqual(cmc_result.cmc_count, 0)
        self.assertGreaterEqual(cmc_result.similarity_score, 0.0)
        self.assertLessEqual(cmc_result.similarity_score, 1.0)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_afte_conclusion(self):
        """Test de determinación de conclusión AFTE"""
        config = create_pipeline_config("forensic")
        pipeline = ScientificPipeline(config)
        
        # Simular resultados de análisis
        similarity_score = 0.85
        quality_score = 0.9
        cmc_count = 8
        consistency_score = 0.8
        
        conclusion = pipeline.determine_afte_conclusion(
            similarity_score, quality_score, cmc_count, consistency_score
        )
        
        self.assertIsInstance(conclusion, AFTEConclusion)
        # Con puntuaciones altas, debería ser identificación
        self.assertEqual(conclusion, AFTEConclusion.IDENTIFICATION)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_afte_conclusion_elimination(self):
        """Test de conclusión AFTE de eliminación"""
        config = create_pipeline_config("forensic")
        pipeline = ScientificPipeline(config)
        
        # Simular resultados de análisis con baja similitud
        similarity_score = 0.15
        quality_score = 0.8
        cmc_count = 1
        consistency_score = 0.2
        
        conclusion = pipeline.determine_afte_conclusion(
            similarity_score, quality_score, cmc_count, consistency_score
        )
        
        self.assertEqual(conclusion, AFTEConclusion.ELIMINATION)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_complete_analysis(self):
        """Test de análisis completo del pipeline"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Realizar análisis completo
        start_time = time.time()
        result = pipeline.analyze(self.image1_path, self.image2_path)
        processing_time = time.time() - start_time
        
        # Verificar resultado
        self.assertIsInstance(result, PipelineResult)
        self.assertIsNotNone(result.afte_conclusion)
        self.assertGreater(result.processing_time, 0.0)
        self.assertLess(processing_time, 60.0)  # No debe tomar más de 1 minuto
        
        # Verificar que todos los componentes se ejecutaron
        self.assertIsNotNone(result.quality_report1)
        self.assertIsNotNone(result.quality_report2)
        self.assertIsNotNone(result.match_result)
        
        if config.cmc_analysis.enabled:
            self.assertIsNotNone(result.cmc_result)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_export_report(self):
        """Test de exportación de reporte"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Crear resultado simulado
        result = PipelineResult(
            afte_conclusion=AFTEConclusion.IDENTIFICATION,
            confidence_score=0.85,
            processing_time=5.2,
            quality_report1=Mock(),
            quality_report2=Mock(),
            match_result=Mock(),
            cmc_result=Mock(),
            metadata={"test": True}
        )
        
        # Exportar reporte
        report_path = os.path.join(self.temp_dir, "test_report.json")
        success = pipeline.export_report(result, report_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(report_path))
        
        # Verificar contenido del reporte
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn("afte_conclusion", report_data)
        self.assertIn("confidence_score", report_data)
        self.assertIn("processing_time", report_data)
        self.assertIn("timestamp", report_data)


class TestPipelineIntegration(unittest.TestCase):
    """Tests de integración para el pipeline completo"""
    
    def setUp(self):
        """Configuración inicial para tests de integración"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear conjunto de imágenes de prueba más realistas
        self.test_images = self._create_realistic_test_images()
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_realistic_test_images(self):
        """Crear imágenes de prueba más realistas para testing de integración"""
        images = {}
        
        # Imagen de casquillo con marcas de percutor
        casquillo = np.zeros((400, 400), dtype=np.uint8)
        center = (200, 200)
        
        # Círculo exterior (borde del casquillo)
        cv2.circle(casquillo, center, 180, 200, 3)
        
        # Marca de percutor (círculo central con textura)
        cv2.circle(casquillo, center, 30, 255, -1)
        
        # Añadir algunas marcas características
        for angle in range(0, 360, 45):
            x = int(center[0] + 25 * np.cos(np.radians(angle)))
            y = int(center[1] + 25 * np.sin(np.radians(angle)))
            cv2.circle(casquillo, (x, y), 3, 100, -1)
        
        # Añadir ruido realista
        noise = np.random.normal(0, 15, casquillo.shape).astype(np.int16)
        casquillo = np.clip(casquillo.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        images['casquillo1'] = casquillo
        
        # Crear casquillo similar (mismo arma)
        casquillo2 = casquillo.copy()
        # Añadir ligera rotación y desplazamiento
        M = cv2.getRotationMatrix2D(center, 5, 1.0)
        casquillo2 = cv2.warpAffine(casquillo2, M, (400, 400))
        # Añadir ruido diferente
        noise2 = np.random.normal(0, 12, casquillo2.shape).astype(np.int16)
        casquillo2 = np.clip(casquillo2.astype(np.int16) + noise2, 0, 255).astype(np.uint8)
        
        images['casquillo2'] = casquillo2
        
        # Crear casquillo diferente (arma diferente)
        casquillo_diff = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(casquillo_diff, center, 180, 200, 3)
        # Marca de percutor cuadrada en lugar de circular
        cv2.rectangle(casquillo_diff, (170, 170), (230, 230), 255, -1)
        
        # Marcas diferentes
        for i in range(8):
            x = int(center[0] + 40 * np.cos(np.radians(i * 45)))
            y = int(center[1] + 40 * np.sin(np.radians(i * 45)))
            cv2.rectangle(casquillo_diff, (x-2, y-2), (x+2, y+2), 150, -1)
        
        noise3 = np.random.normal(0, 18, casquillo_diff.shape).astype(np.int16)
        casquillo_diff = np.clip(casquillo_diff.astype(np.int16) + noise3, 0, 255).astype(np.uint8)
        
        images['casquillo_diferente'] = casquillo_diff
        
        # Guardar imágenes
        for name, img in images.items():
            path = os.path.join(self.temp_dir, f"{name}.jpg")
            cv2.imwrite(path, img)
            images[name] = path
        
        return images
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_similar_images_identification(self):
        """Test de identificación con imágenes similares"""
        config = create_pipeline_config("forensic")
        pipeline = ScientificPipeline(config)
        
        # Analizar casquillos del mismo arma
        result = pipeline.analyze(
            self.test_images['casquillo1'],
            self.test_images['casquillo2']
        )
        
        self.assertIsInstance(result, PipelineResult)
        # Debería identificar como mismo origen
        self.assertIn(result.afte_conclusion, [
            AFTEConclusion.IDENTIFICATION,
            AFTEConclusion.PROBABLE_IDENTIFICATION
        ])
        self.assertGreater(result.confidence_score, 0.5)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_different_images_elimination(self):
        """Test de eliminación con imágenes diferentes"""
        config = create_pipeline_config("forensic")
        pipeline = ScientificPipeline(config)
        
        # Analizar casquillos de armas diferentes
        result = pipeline.analyze(
            self.test_images['casquillo1'],
            self.test_images['casquillo_diferente']
        )
        
        self.assertIsInstance(result, PipelineResult)
        # Debería eliminar como diferentes orígenes
        self.assertIn(result.afte_conclusion, [
            AFTEConclusion.ELIMINATION,
            AFTEConclusion.PROBABLE_ELIMINATION,
            AFTEConclusion.INCONCLUSIVE
        ])
    
    @unittest.skipUnless(PIPELINE_AVAILABLE and COMPONENTS_AVAILABLE, "Componentes no disponibles")
    def test_performance_benchmarks(self):
        """Test de benchmarks de rendimiento"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Medir tiempo de análisis
        start_time = time.time()
        result = pipeline.analyze(
            self.test_images['casquillo1'],
            self.test_images['casquillo2']
        )
        total_time = time.time() - start_time
        
        # Verificar que el análisis se completa en tiempo razonable
        self.assertLess(total_time, 30.0)  # Menos de 30 segundos
        self.assertGreater(result.processing_time, 0.0)
        
        # Verificar que el tiempo reportado es consistente
        self.assertAlmostEqual(result.processing_time, total_time, delta=1.0)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_different_pipeline_levels(self):
        """Test de diferentes niveles de pipeline"""
        levels = ["basic", "standard", "advanced", "forensic"]
        results = {}
        
        for level in levels:
            config = create_pipeline_config(level)
            pipeline = ScientificPipeline(config)
            
            start_time = time.time()
            result = pipeline.analyze(
                self.test_images['casquillo1'],
                self.test_images['casquillo2']
            )
            processing_time = time.time() - start_time
            
            results[level] = {
                'result': result,
                'time': processing_time
            }
            
            self.assertIsInstance(result, PipelineResult)
            self.assertIsNotNone(result.afte_conclusion)
        
        # Verificar que niveles más altos toman más tiempo pero son más precisos
        self.assertLess(results['basic']['time'], results['forensic']['time'])
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_error_handling(self):
        """Test de manejo de errores"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Test con archivos inexistentes
        result = pipeline.analyze("nonexistent1.jpg", "nonexistent2.jpg")
        self.assertIsNone(result)
        
        # Test con archivos corruptos
        corrupt_file = os.path.join(self.temp_dir, "corrupt.jpg")
        with open(corrupt_file, 'wb') as f:
            f.write(b"not an image")
        
        result = pipeline.analyze(corrupt_file, self.test_images['casquillo1'])
        self.assertIsNone(result)


class TestPipelineReporting(unittest.TestCase):
    """Tests para el sistema de reportes del pipeline"""
    
    def setUp(self):
        """Configuración inicial para tests de reportes"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_json_report_export(self):
        """Test de exportación de reporte JSON"""
        config = create_pipeline_config("standard")
        pipeline = ScientificPipeline(config)
        
        # Crear resultado simulado
        result = PipelineResult(
            afte_conclusion=AFTEConclusion.IDENTIFICATION,
            confidence_score=0.85,
            processing_time=5.2,
            quality_report1=Mock(),
            quality_report2=Mock(),
            match_result=Mock(),
            cmc_result=Mock(),
            metadata={"examiner": "Test Examiner", "case_id": "TEST001"}
        )
        
        # Exportar reporte JSON
        json_path = os.path.join(self.temp_dir, "report.json")
        success = pipeline.export_report(result, json_path, format="json")
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_path))
        
        # Verificar estructura del JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        required_fields = [
            "afte_conclusion", "confidence_score", "processing_time",
            "timestamp", "pipeline_version", "metadata"
        ]
        
        for field in required_fields:
            self.assertIn(field, data)
    
    @unittest.skipUnless(PIPELINE_AVAILABLE, "Pipeline no disponible")
    def test_detailed_report_content(self):
        """Test de contenido detallado del reporte"""
        config = create_pipeline_config("forensic")
        config.export_detailed_report = True
        pipeline = ScientificPipeline(config)
        
        # Crear resultado con datos detallados
        result = PipelineResult(
            afte_conclusion=AFTEConclusion.IDENTIFICATION,
            confidence_score=0.92,
            processing_time=8.7,
            quality_report1=Mock(overall_quality=0.85, snr=25.3),
            quality_report2=Mock(overall_quality=0.88, snr=27.1),
            match_result=Mock(similarity_score=0.89, num_matches=156),
            cmc_result=Mock(cmc_count=12, similarity_score=0.91),
            metadata={
                "examiner": "Dr. Test",
                "case_id": "CASE2024001",
                "evidence_id1": "EV001",
                "evidence_id2": "EV002"
            }
        )
        
        # Exportar reporte detallado
        report_path = os.path.join(self.temp_dir, "detailed_report.json")
        success = pipeline.export_report(result, report_path, format="json")
        
        self.assertTrue(success)
        
        # Verificar contenido detallado
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data["afte_conclusion"], "IDENTIFICATION")
        self.assertEqual(data["confidence_score"], 0.92)
        self.assertIn("quality_assessment", data)
        self.assertIn("feature_matching", data)
        self.assertIn("cmc_analysis", data)


if __name__ == '__main__':
    # Configurar logging para los tests
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar tests
    unittest.main(verbosity=2)