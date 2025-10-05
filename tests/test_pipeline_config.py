#!/usr/bin/env python3
"""
Tests para la Configuración del Pipeline Científico
==================================================

Este módulo contiene tests específicos para validar la configuración
del pipeline científico, incluyendo diferentes niveles de análisis,
configuraciones predefinidas y serialización.
"""

import unittest
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos de configuración
try:
    from core.pipeline_config import (
        PipelineConfiguration, PipelineLevel,
        QualityAssessmentConfig, PreprocessingConfig, ROIDetectionConfig,
        MatchingConfig, CMCAnalysisConfig, AFTEConclusionConfig,
        create_pipeline_config, get_available_levels, get_level_description,
        get_recommended_level, get_predefined_config, list_predefined_configs,
        PREDEFINED_CONFIGS
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Error importando configuración: {e}")
    CONFIG_AVAILABLE = False

# Importar enums necesarios
try:
    from image_processing.unified_preprocessor import PreprocessingLevel
    from image_processing.unified_roi_detector import DetectionLevel
    from matching.unified_matcher import AlgorithmType
    ENUMS_AVAILABLE = True
except ImportError as e:
    print(f"Error importando enums: {e}")
    ENUMS_AVAILABLE = False


class TestPipelineLevel(unittest.TestCase):
    """Tests para los niveles de pipeline"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_pipeline_levels_exist(self):
        """Test que todos los niveles de pipeline existen"""
        levels = [level.value for level in PipelineLevel]
        
        expected_levels = ["basic", "standard", "advanced", "forensic"]
        for level in expected_levels:
            self.assertIn(level, levels)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_get_available_levels(self):
        """Test de obtención de niveles disponibles"""
        levels = get_available_levels()
        
        self.assertIsInstance(levels, list)
        self.assertEqual(len(levels), 4)
        self.assertIn("basic", levels)
        self.assertIn("standard", levels)
        self.assertIn("advanced", levels)
        self.assertIn("forensic", levels)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_descriptions(self):
        """Test de descripciones de niveles"""
        for level in get_available_levels():
            description = get_level_description(level)
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 10)
            self.assertNotEqual(description, "Nivel no reconocido")
        
        # Test nivel inválido
        invalid_description = get_level_description("invalid_level")
        self.assertEqual(invalid_description, "Nivel no reconocido")


class TestQualityAssessmentConfig(unittest.TestCase):
    """Tests para configuración de evaluación de calidad"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = QualityAssessmentConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.min_snr, 20.0)
        self.assertEqual(config.min_contrast, 0.3)
        self.assertEqual(config.min_uniformity, 0.7)
        self.assertEqual(config.min_sharpness, 0.5)
        self.assertEqual(config.min_resolution, 300.0)
        self.assertFalse(config.strict_mode)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_thresholds(self):
        """Test de umbrales por nivel"""
        config = QualityAssessmentConfig()
        
        # Verificar que todos los niveles tienen configuración
        for level in ["basic", "standard", "advanced", "forensic"]:
            self.assertIn(level, config.level_thresholds)
            thresholds = config.level_thresholds[level]
            
            # Verificar campos requeridos
            required_fields = ["min_snr", "min_contrast", "min_uniformity", 
                             "min_sharpness", "min_resolution"]
            for field in required_fields:
                self.assertIn(field, thresholds)
                self.assertIsInstance(thresholds[field], (int, float))
                self.assertGreater(thresholds[field], 0)
        
        # Verificar que forensic tiene umbrales más estrictos que basic
        basic_snr = config.level_thresholds["basic"]["min_snr"]
        forensic_snr = config.level_thresholds["forensic"]["min_snr"]
        self.assertGreater(forensic_snr, basic_snr)


class TestPreprocessingConfig(unittest.TestCase):
    """Tests para configuración de preprocesamiento"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE and ENUMS_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = PreprocessingConfig()
        
        self.assertEqual(config.level, PreprocessingLevel.STANDARD)
        self.assertTrue(config.enable_noise_reduction)
        self.assertTrue(config.enable_contrast_enhancement)
        self.assertTrue(config.enable_sharpening)
        self.assertTrue(config.enable_normalization)
        
        # Verificar parámetros específicos
        self.assertEqual(config.gaussian_kernel_size, 5)
        self.assertEqual(config.gaussian_sigma, 1.0)
        self.assertEqual(config.clahe_clip_limit, 2.0)
        self.assertEqual(config.clahe_tile_grid_size, (8, 8))
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_configs(self):
        """Test de configuraciones por nivel"""
        config = PreprocessingConfig()
        
        for level in ["basic", "standard", "advanced", "forensic"]:
            self.assertIn(level, config.level_configs)
            level_config = config.level_configs[level]
            
            # Verificar campos básicos
            self.assertIn("enable_noise_reduction", level_config)
            self.assertIn("enable_normalization", level_config)
            self.assertIsInstance(level_config["enable_noise_reduction"], bool)
            self.assertIsInstance(level_config["enable_normalization"], bool)


class TestROIDetectionConfig(unittest.TestCase):
    """Tests para configuración de detección de ROI"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE and ENUMS_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = ROIDetectionConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.detection_level, DetectionLevel.STANDARD)
        self.assertEqual(config.min_roi_area, 1000)
        self.assertEqual(config.max_roi_count, 5)
        
        # Verificar parámetros de watershed
        self.assertEqual(config.watershed_markers, 10)
        self.assertEqual(config.watershed_compactness, 0.1)
        self.assertEqual(config.watershed_sigma, 1.0)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_progression(self):
        """Test de progresión de configuraciones por nivel"""
        config = ROIDetectionConfig()
        
        # Verificar que forensic permite más ROIs que basic
        basic_config = config.level_configs["basic"]
        forensic_config = config.level_configs["forensic"]
        
        self.assertGreater(forensic_config["max_roi_count"], basic_config["max_roi_count"])
        self.assertLess(forensic_config["min_roi_area"], basic_config["min_roi_area"])


class TestMatchingConfig(unittest.TestCase):
    """Tests para configuración de matching"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE and ENUMS_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = MatchingConfig()
        
        self.assertEqual(config.algorithm, AlgorithmType.ORB)
        self.assertEqual(config.similarity_threshold, 0.7)
        self.assertEqual(config.max_features, 5000)
        self.assertTrue(config.enable_ransac)
        self.assertEqual(config.ransac_threshold, 5.0)
        self.assertEqual(config.ransac_max_trials, 1000)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_algorithm_parameters(self):
        """Test de parámetros específicos por algoritmo"""
        config = MatchingConfig()
        
        # Parámetros ORB
        self.assertEqual(config.orb_n_features, 5000)
        self.assertEqual(config.orb_scale_factor, 1.2)
        self.assertEqual(config.orb_n_levels, 8)
        
        # Parámetros SIFT
        self.assertEqual(config.sift_n_features, 0)  # Sin límite
        self.assertEqual(config.sift_n_octave_layers, 3)
        self.assertEqual(config.sift_contrast_threshold, 0.04)
        self.assertEqual(config.sift_edge_threshold, 10.0)
        self.assertEqual(config.sift_sigma, 1.6)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_algorithm_selection(self):
        """Test de selección de algoritmo por nivel"""
        config = MatchingConfig()
        
        # Basic y Standard usan ORB
        self.assertEqual(config.level_configs["basic"]["algorithm"], AlgorithmType.ORB)
        self.assertEqual(config.level_configs["standard"]["algorithm"], AlgorithmType.ORB)
        
        # Advanced y Forensic usan SIFT
        self.assertEqual(config.level_configs["advanced"]["algorithm"], AlgorithmType.SIFT)
        self.assertEqual(config.level_configs["forensic"]["algorithm"], AlgorithmType.SIFT)


class TestCMCAnalysisConfig(unittest.TestCase):
    """Tests para configuración de análisis CMC"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = CMCAnalysisConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.cmc_threshold, 6)
        self.assertEqual(config.min_cell_size, 10)
        self.assertEqual(config.max_cell_size, 100)
        self.assertEqual(config.overlap_threshold, 0.3)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_threshold_progression(self):
        """Test de progresión de umbrales por nivel"""
        config = CMCAnalysisConfig()
        
        # Verificar que forensic tiene umbral más alto que basic
        basic_threshold = config.level_configs["basic"]["cmc_threshold"]
        forensic_threshold = config.level_configs["forensic"]["cmc_threshold"]
        
        self.assertGreaterEqual(forensic_threshold, basic_threshold)


class TestAFTEConclusionConfig(unittest.TestCase):
    """Tests para configuración de conclusiones AFTE"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_default_values(self):
        """Test de valores por defecto"""
        config = AFTEConclusionConfig()
        
        self.assertEqual(config.identification_threshold, 0.8)
        self.assertEqual(config.elimination_threshold, 0.3)
        
        # Verificar pesos
        self.assertEqual(config.similarity_weight, 0.4)
        self.assertEqual(config.quality_weight, 0.2)
        self.assertEqual(config.cmc_weight, 0.3)
        self.assertEqual(config.consistency_weight, 0.1)
        
        # Verificar que los pesos suman 1.0
        total_weight = (config.similarity_weight + config.quality_weight + 
                       config.cmc_weight + config.consistency_weight)
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_threshold_consistency(self):
        """Test de consistencia de umbrales"""
        config = AFTEConclusionConfig()
        
        # El umbral de identificación debe ser mayor que el de eliminación
        self.assertGreater(config.identification_threshold, config.elimination_threshold)
        
        # Verificar para todos los niveles
        for level in ["basic", "standard", "advanced", "forensic"]:
            level_config = config.level_configs[level]
            id_threshold = level_config["identification_threshold"]
            elim_threshold = level_config["elimination_threshold"]
            
            self.assertGreater(id_threshold, elim_threshold)
            self.assertGreaterEqual(id_threshold, 0.0)
            self.assertLessEqual(id_threshold, 1.0)
            self.assertGreaterEqual(elim_threshold, 0.0)
            self.assertLessEqual(elim_threshold, 1.0)


class TestPipelineConfiguration(unittest.TestCase):
    """Tests para la configuración completa del pipeline"""
    
    def setUp(self):
        """Configuración inicial para los tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Limpieza después de los tests"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_default_configuration(self):
        """Test de configuración por defecto"""
        config = PipelineConfiguration()
        
        self.assertEqual(config.level, PipelineLevel.STANDARD)
        self.assertIsNotNone(config.quality_assessment)
        self.assertIsNotNone(config.preprocessing)
        self.assertIsNotNone(config.roi_detection)
        self.assertIsNotNone(config.matching)
        self.assertIsNotNone(config.cmc_analysis)
        self.assertIsNotNone(config.afte_conclusion)
        
        # Configuraciones generales
        self.assertTrue(config.enable_parallel_processing)
        self.assertEqual(config.max_processing_threads, 4)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.cache_directory, "cache/pipeline")
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_apply_level_configuration(self):
        """Test de aplicación de configuración por nivel"""
        config = PipelineConfiguration()
        
        # Aplicar configuración forensic
        config.apply_level_configuration("forensic")
        
        # Verificar que se aplicaron los cambios
        self.assertGreaterEqual(config.quality_assessment.min_snr, 30.0)
        self.assertGreaterEqual(config.afte_conclusion.identification_threshold, 0.9)
        
        # Aplicar configuración basic
        config.apply_level_configuration("basic")
        
        # Verificar que se aplicaron los cambios básicos
        self.assertLessEqual(config.quality_assessment.min_snr, 15.0)
        self.assertLessEqual(config.afte_conclusion.identification_threshold, 0.7)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_to_dict_conversion(self):
        """Test de conversión a diccionario"""
        config = PipelineConfiguration()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        
        # Verificar campos principales
        required_fields = [
            "level", "quality_assessment", "preprocessing", 
            "roi_detection", "matching", "cmc_analysis", "afte_conclusion"
        ]
        
        for field in required_fields:
            self.assertIn(field, config_dict)
            self.assertIsInstance(config_dict[field], dict)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_from_dict_creation(self):
        """Test de creación desde diccionario"""
        # Crear configuración original
        original_config = PipelineConfiguration()
        original_config.apply_level_configuration("advanced")
        
        # Convertir a diccionario
        config_dict = original_config.to_dict()
        
        # Recrear desde diccionario
        restored_config = PipelineConfiguration.from_dict(config_dict)
        
        # Verificar que se restauró correctamente
        self.assertEqual(restored_config.level.value, "advanced")
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_serialization_roundtrip(self):
        """Test de serialización completa (ida y vuelta)"""
        # Crear configuración
        config = create_pipeline_config("forensic")
        
        # Serializar a JSON
        json_path = os.path.join(self.temp_dir, "config.json")
        config_dict = config.to_dict()
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Deserializar desde JSON
        with open(json_path, 'r') as f:
            loaded_dict = json.load(f)
        
        restored_config = PipelineConfiguration.from_dict(loaded_dict)
        
        # Verificar que la configuración se preservó
        self.assertEqual(restored_config.level.value, "forensic")


class TestPipelineConfigurationFactory(unittest.TestCase):
    """Tests para las funciones de factory de configuración"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_create_pipeline_config(self):
        """Test de creación de configuración de pipeline"""
        for level in get_available_levels():
            config = create_pipeline_config(level)
            
            self.assertIsInstance(config, PipelineConfiguration)
            self.assertEqual(config.level.value, level)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_get_recommended_level(self):
        """Test de recomendación de nivel"""
        # Imagen de alta calidad, caso simple
        level = get_recommended_level(0.9, "low")
        self.assertEqual(level, "advanced")
        
        # Imagen de baja calidad
        level = get_recommended_level(0.3, "medium")
        self.assertEqual(level, "basic")
        
        # Caso forense
        level = get_recommended_level(0.8, "forensic")
        self.assertEqual(level, "forensic")
        
        # Imagen media, caso complejo
        level = get_recommended_level(0.6, "high")
        self.assertEqual(level, "forensic")
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_predefined_configs(self):
        """Test de configuraciones predefinidas"""
        predefined_names = list_predefined_configs()
        
        self.assertIsInstance(predefined_names, list)
        self.assertGreater(len(predefined_names), 0)
        
        # Verificar configuraciones específicas
        expected_configs = ["quick_screening", "standard_comparison", 
                          "forensic_analysis", "research_mode"]
        
        for config_name in expected_configs:
            self.assertIn(config_name, predefined_names)
            
            # Verificar que se puede crear la configuración
            config = get_predefined_config(config_name)
            self.assertIsInstance(config, PipelineConfiguration)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_predefined_config_properties(self):
        """Test de propiedades específicas de configuraciones predefinidas"""
        # Quick screening - debe ser rápido
        quick_config = get_predefined_config("quick_screening")
        self.assertEqual(quick_config.level.value, "basic")
        self.assertFalse(quick_config.quality_assessment.enabled)
        
        # Forensic analysis - debe ser exhaustivo
        forensic_config = get_predefined_config("forensic_analysis")
        self.assertEqual(forensic_config.level.value, "forensic")
        self.assertTrue(forensic_config.quality_assessment.strict_mode)
        self.assertTrue(forensic_config.export_intermediate_results)
        
        # Research mode - debe tener todas las opciones habilitadas
        research_config = get_predefined_config("research_mode")
        self.assertTrue(research_config.export_intermediate_results)
        self.assertTrue(research_config.export_visualizations)
        self.assertTrue(research_config.enable_parallel_processing)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_invalid_predefined_config(self):
        """Test de configuración predefinida inválida"""
        with self.assertRaises(ValueError):
            get_predefined_config("nonexistent_config")


class TestConfigurationValidation(unittest.TestCase):
    """Tests para validación de configuraciones"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_valid_configurations(self):
        """Test de configuraciones válidas"""
        for level in get_available_levels():
            config = create_pipeline_config(level)
            
            # Verificar rangos válidos
            self.assertGreaterEqual(config.quality_assessment.min_snr, 0.0)
            self.assertGreaterEqual(config.quality_assessment.min_contrast, 0.0)
            self.assertLessEqual(config.quality_assessment.min_contrast, 1.0)
            
            self.assertGreaterEqual(config.afte_conclusion.identification_threshold, 0.0)
            self.assertLessEqual(config.afte_conclusion.identification_threshold, 1.0)
            
            self.assertGreater(config.matching.max_features, 0)
            self.assertGreater(config.cmc_analysis.cmc_threshold, 0)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_configuration_consistency(self):
        """Test de consistencia entre configuraciones"""
        for level in get_available_levels():
            config = create_pipeline_config(level)
            
            # El umbral de identificación debe ser mayor que el de eliminación
            self.assertGreater(
                config.afte_conclusion.identification_threshold,
                config.afte_conclusion.elimination_threshold
            )
            
            # Los tamaños de celda CMC deben ser consistentes
            self.assertLess(
                config.cmc_analysis.min_cell_size,
                config.cmc_analysis.max_cell_size
            )
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "Configuración no disponible")
    def test_level_progression(self):
        """Test de progresión lógica entre niveles"""
        basic_config = create_pipeline_config("basic")
        forensic_config = create_pipeline_config("forensic")
        
        # Forensic debe tener umbrales más estrictos
        self.assertGreater(
            forensic_config.quality_assessment.min_snr,
            basic_config.quality_assessment.min_snr
        )
        
        self.assertGreater(
            forensic_config.afte_conclusion.identification_threshold,
            basic_config.afte_conclusion.identification_threshold
        )
        
        # Forensic debe permitir más características
        self.assertGreaterEqual(
            forensic_config.matching.max_features,
            basic_config.matching.max_features
        )


if __name__ == '__main__':
    # Configurar logging para los tests
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar tests
    unittest.main(verbosity=2)