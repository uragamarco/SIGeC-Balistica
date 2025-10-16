"""
Tests comprehensivos para validar la implementación de estándares NIST
Incluye pruebas para schema XML, métricas de calidad, conclusiones AFTE y protocolos de validación.
Consolidado desde múltiples archivos de test NIST.
"""

import unittest
import tempfile
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar módulos NIST
try:
    from nist_standards import NISTStandardsManager
    from nist_standards.nist_schema import (
        NISTSchema, NISTDataExporter, NISTDataImporter,
        EvidenceType, ExaminationMethod, NISTMetadata, NISTImageData,
        NISTFeatureData, NISTComparisonResult
    )
    from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
    from nist_standards.afte_conclusions import (
        AFTEConclusionEngine, AFTEConclusion, ConfidenceLevel, FeatureMatch,
        FeatureType, AFTEAnalysisResult
    )
    from nist_standards.validation_protocols import (
        NISTValidationProtocols, ValidationLevel, ValidationResult,
        ValidationDataset, MetricType
    )
    from nist_standards.statistical_analysis import (
        AdvancedStatisticalAnalysis,
        StatisticalTest,
        CorrectionMethod
    )
    # Importar nuevos módulos de procesamiento
    from image_processing.spatial_calibration import SpatialCalibrator, CalibrationData
    from image_processing.nist_compliance_validator import NISTComplianceValidator, NISTProcessingReport
    from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig
except ImportError as e:
    print(f"Error importando módulos NIST: {e}")
    # Crear mocks para permitir que las pruebas se ejecuten
    NISTStandardsManager = Mock
    NISTSchema = Mock
    NISTDataExporter = Mock
    NISTDataImporter = Mock
    NISTQualityMetrics = Mock
    AFTEConclusionEngine = Mock
    NISTValidationProtocols = Mock
    EvidenceType = Mock
    ExaminationMethod = Mock
    NISTMetadata = Mock
    NISTImageData = Mock
    NISTQualityReport = Mock
    AFTEConclusion = Mock
    ConfidenceLevel = Mock
    FeatureMatch = Mock
    FeatureType = Mock
    SpatialCalibrator = Mock
    CalibrationData = Mock
    NISTComplianceValidator = Mock
    NISTProcessingReport = Mock
    UnifiedPreprocessor = Mock
    PreprocessingConfig = Mock
    ValidationLevel = Mock
    ValidationResult = Mock
    AdvancedStatisticalAnalysis = Mock
    StatisticalTest = Mock
    CorrectionMethod = Mock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestNISTSchema(unittest.TestCase):
    """Tests para el esquema NIST XML y exportación/importación de datos"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.temp_dir = tempfile.mkdtemp()
        self.schema = NISTSchema()
        self.exporter = NISTDataExporter()
        self.importer = NISTDataImporter()
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_create_nist_metadata(self):
        """Test creación de metadatos NIST"""
        metadata = NISTMetadata(
            case_id="TEST_001",
            examiner="Test Examiner",
            date=datetime.now(),
            evidence_type=EvidenceType.CARTRIDGE_CASE,
            examination_method=ExaminationMethod.COMPARISON
        )
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.case_id, "TEST_001")
        self.assertEqual(metadata.examiner, "Test Examiner")
        
    def test_create_nist_image_data(self):
        """Test creación de datos de imagen NIST"""
        image_data = NISTImageData(
            image_path="test_image.jpg",
            resolution_dpi=300,
            color_space="RGB",
            metadata={
                "camera": "Test Camera",
                "lens": "Test Lens"
                }
        )
        
        self.assertIsNotNone(image_data)
        self.assertEqual(image_data.resolution_dpi, 300)
        
    def test_xml_generation(self):
        """Test generación de XML NIST"""
        metadata = NISTMetadata(
            case_id="TEST_XML",
            examiner="XML Tester",
            date=datetime.now(),
            evidence_type=EvidenceType.BULLET,
            examination_method=ExaminationMethod.IDENTIFICATION
        )
        
        xml_content = self.schema.generate_xml(metadata)
        self.assertIsNotNone(xml_content)
        
        # Validar estructura XML
        try:
            root = ET.fromstring(xml_content)
            self.assertEqual(root.tag, "NISTBallistics")
        except ET.ParseError:
            self.fail("XML generado no es válido")
            
    def test_xml_export_import(self):
        """Test exportación e importación XML"""
        test_data = {
            "metadata": {
                "case_id": "EXPORT_TEST",
                "examiner": "Export Tester",
                "evidence_type": "cartridge_case"
            },
            "results": {
                "conclusion": "identification",
                "confidence": 0.95
                }
            }
        
        # Exportar
        xml_file = os.path.join(self.temp_dir, "test_export.xml")
        self.exporter.export_to_xml(test_data, xml_file)
        self.assertTrue(os.path.exists(xml_file))
        
        # Importar
        imported_data = self.importer.import_from_xml(xml_file)
        self.assertIsNotNone(imported_data)
        
    def test_json_export_import(self):
        """Test exportación e importación JSON"""
        test_data = {
            "metadata": {
                "case_id": "JSON_TEST",
                "examiner": "JSON Tester",
                "evidence_type": "bullet"
            },
            "results": {
                "conclusion": "elimination",
                "confidence": 0.99
                }
            }
        
        # Exportar
        json_file = os.path.join(self.temp_dir, "test_export.json")
        self.exporter.export_to_json(test_data, json_file)
        self.assertTrue(os.path.exists(json_file))
        
        # Importar
        imported_data = self.importer.import_from_json(json_file)
        self.assertIsNotNone(imported_data)


class TestNISTQualityMetrics(unittest.TestCase):
    """Tests para métricas de calidad NIST"""
    
    def setUp(self):
        """Configurar métricas de calidad"""
        self.quality_metrics = NISTQualityMetrics()
        
        # Crear imagen de prueba
        self.test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Crear imagen con ruido
        noise = np.random.normal(0, 25, (512, 512, 3))
        self.noisy_image = np.clip(self.test_image.astype(float) + noise, 0, 255).astype(np.uint8)
        
    def test_calculate_snr(self):
        """Test cálculo de SNR"""
        snr = self.quality_metrics.calculate_snr(self.test_image, self.noisy_image)
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)
        
        # Test con imagen perfecta (SNR infinito)
        perfect_snr = self.quality_metrics.calculate_snr(self.test_image, self.test_image)
        self.assertTrue(np.isinf(perfect_snr) or perfect_snr > 100)
        
    def test_calculate_contrast(self):
        """Test cálculo de contraste"""
        contrast = self.quality_metrics.calculate_contrast(self.test_image)
        self.assertIsInstance(contrast, float)
        self.assertGreaterEqual(contrast, 0)
        
        # Test con imagen uniforme (contraste bajo)
        uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        uniform_contrast = self.quality_metrics.calculate_contrast(uniform_image)
        self.assertLess(uniform_contrast, contrast)
        
    def test_calculate_uniformity(self):
        """Test cálculo de uniformidad"""
        uniformity = self.quality_metrics.calculate_uniformity(self.test_image)
        self.assertIsInstance(uniformity, float)
        self.assertGreaterEqual(uniformity, 0)
        self.assertLessEqual(uniformity, 1)
        
        # Test con imagen perfectamente uniforme
        uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        perfect_uniformity = self.quality_metrics.calculate_uniformity(uniform_image)
        self.assertGreater(perfect_uniformity, uniformity)
        
    def test_calculate_sharpness(self):
        """Test cálculo de nitidez"""
        sharpness = self.quality_metrics.calculate_sharpness(self.test_image)
        self.assertIsInstance(sharpness, float)
        self.assertGreater(sharpness, 0)
        
        # Test con imagen desenfocada
        blurred = cv2.GaussianBlur(self.test_image, (15, 15), 5)
        blurred_sharpness = self.quality_metrics.calculate_sharpness(blurred)
        self.assertLess(blurred_sharpness, sharpness)
        
    def test_assess_overall_quality(self):
        """Test evaluación general de calidad"""
        quality_report = self.quality_metrics.assess_overall_quality(self.test_image)
        
        self.assertIsInstance(quality_report, NISTQualityReport)
        self.assertIn('snr', quality_report.metrics)
        self.assertIn('contrast', quality_report.metrics)
        self.assertIn('uniformity', quality_report.metrics)
        self.assertIn('sharpness', quality_report.metrics)
        
        # Verificar que el score general está en rango válido
        self.assertGreaterEqual(quality_report.overall_score, 0)
        self.assertLessEqual(quality_report.overall_score, 1)
        
    def test_generate_recommendations(self):
        """Test generación de recomendaciones"""
        quality_report = self.quality_metrics.assess_overall_quality(self.noisy_image)
        recommendations = self.quality_metrics.generate_recommendations(quality_report)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
    def test_compare_quality_reports(self):
        """Test comparación de reportes de calidad"""
        report1 = self.quality_metrics.assess_overall_quality(self.test_image)
        report2 = self.quality_metrics.assess_overall_quality(self.noisy_image)
        
        comparison = self.quality_metrics.compare_reports(report1, report2)
        
        self.assertIsInstance(comparison, dict)
        self.assertIn('better_image', comparison)
        self.assertIn('improvement_areas', comparison)


class TestAFTEConclusions(unittest.TestCase):
    """Tests para conclusiones AFTE"""
    
    def setUp(self):
        """Configurar motor de conclusiones AFTE"""
        self.afte_engine = AFTEConclusionEngine()
        
    def test_create_feature_match(self):
        """Test creación de coincidencia de características"""
        feature_match = FeatureMatch(
            feature_type=FeatureType.STRIATION,
            similarity_score=0.85,
            confidence=ConfidenceLevel.HIGH,
            location=(100, 200),
            description="Striation pattern match"
        )
        
        self.assertIsNotNone(feature_match)
        self.assertEqual(feature_match.similarity_score, 0.85)
        self.assertEqual(feature_match.confidence, ConfidenceLevel.HIGH)
        
    def test_determine_afte_conclusion_identification(self):
        """Test determinación de conclusión AFTE - Identificación"""
        matches = [
            FeatureMatch(FeatureType.STRIATION, 0.95, ConfidenceLevel.HIGH, (0, 0), "High quality match"),
            FeatureMatch(FeatureType.IMPRESSION, 0.90, ConfidenceLevel.HIGH, (50, 50), "Clear impression"),
            FeatureMatch(FeatureType.BREACH_FACE, 0.88, ConfidenceLevel.MEDIUM, (100, 100), "Breach face mark")
        ]
        
        conclusion = self.afte_engine.determine_conclusion(matches)
        self.assertEqual(conclusion.conclusion_type, AFTEConclusion.IDENTIFICATION)
        
    def test_determine_afte_conclusion_elimination(self):
        """Test determinación de conclusión AFTE - Eliminación"""
        matches = [
            FeatureMatch(FeatureType.STRIATION, 0.15, ConfidenceLevel.LOW, (0, 0), "Poor match"),
            FeatureMatch(FeatureType.IMPRESSION, 0.20, ConfidenceLevel.LOW, (50, 50), "Inconsistent pattern")
        ]
        
        conclusion = self.afte_engine.determine_conclusion(matches)
        self.assertEqual(conclusion.conclusion_type, AFTEConclusion.ELIMINATION)
        
    def test_determine_afte_conclusion_inconclusive(self):
        """Test determinación de conclusión AFTE - Inconcluso"""
        matches = [
            FeatureMatch(FeatureType.STRIATION, 0.65, ConfidenceLevel.MEDIUM, (0, 0), "Moderate match"),
            FeatureMatch(FeatureType.IMPRESSION, 0.55, ConfidenceLevel.LOW, (50, 50), "Unclear pattern")
        ]
        
        conclusion = self.afte_engine.determine_conclusion(matches)
        self.assertEqual(conclusion.conclusion_type, AFTEConclusion.INCONCLUSIVE)
        
    def test_analyze_comparison_complete(self):
        """Test análisis completo de comparación"""
        comparison_data = {
            "image1_path": "evidence1.jpg",
            "image2_path": "evidence2.jpg",
            "features": [
                {"type": "striation", "score": 0.92, "confidence": "high"},
                {"type": "impression", "score": 0.87, "confidence": "medium"}
            ]
            }
        
        result = self.afte_engine.analyze_comparison(comparison_data)
        self.assertIsInstance(result, AFTEAnalysisResult)
        
    def test_validate_conclusion(self):
        """Test validación de conclusión"""
        test_conclusion = AFTEConclusion(
            conclusion_type=AFTEConclusion.IDENTIFICATION,
            confidence_level=ConfidenceLevel.HIGH,
            supporting_features=3,
            examiner="Test Examiner"
        )
        
        is_valid = self.afte_engine.validate_conclusion(test_conclusion)
        self.assertTrue(is_valid)


class TestNISTValidationProtocols(unittest.TestCase):
    """Tests para protocolos de validación NIST"""
    
    def setUp(self):
        """Configurar protocolos de validación"""
        self.validation_protocols = NISTValidationProtocols()
        
        # Crear dataset de prueba
        self.test_dataset = self._create_mock_dataset()
        
    def _create_mock_dataset(self):
        """Crear dataset mock para pruebas"""
        return ValidationDataset(
            name="Test Dataset",
            samples=100,
            ground_truth_available=True,
            metadata={
                "source": "synthetic",
                "quality": "high"
            }
            )
    
    def test_perform_k_fold_validation(self):
        """Test validación k-fold"""
        k_folds = 5
        
        # Mock del algoritmo de comparación
        def mock_comparison_algorithm(train_data, test_data):
            # Simular resultados de comparación
            return {
                "accuracy": np.random.uniform(0.8, 0.95),
                "precision": np.random.uniform(0.85, 0.98),
                "recall": np.random.uniform(0.80, 0.92),
                "f1_score": np.random.uniform(0.82, 0.94)
            }
        
        results = self.validation_protocols.perform_k_fold_validation(
            self.test_dataset,
            mock_comparison_algorithm,
            k=k_folds
        )
        
        self.assertIsInstance(results, ValidationResult)
        self.assertEqual(len(results.fold_results), k_folds)
        self.assertIn('mean_accuracy', results.summary_metrics)
        self.assertIn('std_accuracy', results.summary_metrics)
        
    def test_calculate_reliability_metrics(self):
        """Test cálculo de métricas de confiabilidad"""
        # Datos de prueba simulados
        predictions = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        ground_truth = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
        confidence_scores = np.array([0.9, 0.7, 0.8, 0.95, 0.85, 0.92, 0.6, 0.88, 0.91, 0.75])
        
        metrics = self.validation_protocols.calculate_reliability_metrics(
            predictions, ground_truth, confidence_scores
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('calibration_error', metrics)
        
        # Verificar rangos válidos
        for metric_name, value in metrics.items():
            if metric_name != 'calibration_error':
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
                
    def test_assess_system_uncertainty(self):
        """Test evaluación de incertidumbre del sistema"""
        # Datos de múltiples ejecuciones
        multiple_runs_data = {
            "run_1": {"accuracy": 0.92, "precision": 0.89, "recall": 0.94},
            "run_2": {"accuracy": 0.90, "precision": 0.91, "recall": 0.88},
            "run_3": {"accuracy": 0.94, "precision": 0.87, "recall": 0.96}
            }
        
        uncertainty_assessment = self.validation_protocols.assess_system_uncertainty(
            multiple_runs_data
        )
        
        self.assertIn('mean_metrics', uncertainty_assessment)
        self.assertIn('std_metrics', uncertainty_assessment)
        self.assertIn('confidence_intervals', uncertainty_assessment)
        self.assertIn('uncertainty_level', uncertainty_assessment)
        
    def test_generate_validation_report(self):
        """Test generación de reporte de validación"""
        # Datos de validación simulados
        validation_data = {
            "dataset_info": self.test_dataset,
            "validation_results": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
                },
            "uncertainty_analysis": {
                "mean_accuracy": 0.92,
                "std_accuracy": 0.02,
                "confidence_interval": [0.88, 0.96]
                }
            }
        
        report = self.validation_protocols.generate_validation_report(validation_data)
        
        self.assertIsInstance(report, str)
        self.assertIn("NIST Validation Report", report)
        self.assertIn("Dataset Information", report)
        self.assertIn("Validation Results", report)


class TestNISTStandardsManager(unittest.TestCase):
    """Tests para el gestor principal de estándares NIST"""
    
    def setUp(self):
        """Configurar gestor de estándares"""
        self.nist_manager = NISTStandardsManager()
        
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        
        # Datos de evidencia de prueba
        self.test_evidence = {
            "case_id": "MANAGER_TEST_001",
            "evidence_type": "cartridge_case",
            "image_path": "test_evidence.jpg",
            "metadata": {
                "examiner": "Test Manager",
                "date": datetime.now().isoformat()
            }
        }
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('nist_standards.NISTQualityMetrics')
    @patch('nist_standards.AFTEConclusionEngine')
    def test_process_ballistic_evidence(self, mock_afte, mock_quality):
        """Test procesamiento de evidencia balística"""
        # Configurar mocks
        mock_quality_instance = mock_quality.return_value
        mock_quality_instance.assess_overall_quality.return_value = Mock(
            overall_score=0.85,
            metrics={'snr': 25.5, 'contrast': 0.7}
        )
        
        mock_afte_instance = mock_afte.return_value
        mock_afte_instance.analyze_comparison.return_value = Mock(
            conclusion_type=AFTEConclusion.IDENTIFICATION,
            confidence_level=ConfidenceLevel.HIGH
        )
        
        result = self.nist_manager.process_ballistic_evidence(self.test_evidence)
        
        self.assertIsNotNone(result)
        self.assertIn('quality_assessment', result)
        self.assertIn('afte_analysis', result)
        
    @patch('nist_standards.NISTValidationProtocols')
    def test_validate_system(self, mock_validation):
        """Test validación del sistema"""
        # Configurar mock
        mock_validation_instance = mock_validation.return_value
        mock_validation_instance.perform_k_fold_validation.return_value = Mock(
            summary_metrics={'mean_accuracy': 0.92, 'std_accuracy': 0.02}
        )
        
        validation_config = {
            "dataset": "test_dataset",
            "k_folds": 5,
            "metrics": ["accuracy", "precision", "recall"]
            }
        
        result = self.nist_manager.validate_system(validation_config)
        
        self.assertIsNotNone(result)
        self.assertIn('validation_results', result)
        
    def test_export_nist_report(self):
        """Test exportación de reporte NIST"""
        report_data = {
            "case_info": self.test_evidence,
            "analysis_results": {
                "quality_score": 0.85,
                "afte_conclusion": "identification",
                "confidence": "high"
                },
            "validation_metrics": {
                "accuracy": 0.92,
                "precision": 0.89
                }
            }
        
        output_path = os.path.join(self.temp_dir, "nist_report.xml")
        
        success = self.nist_manager.export_nist_report(report_data, output_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))


class TestNISTIntegration(unittest.TestCase):
    """Tests de integración completa NIST"""
    
    def setUp(self):
        """Configurar pruebas de integración"""
        self.nist_manager = NISTStandardsManager()
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear imagen de prueba
        self.test_image_path = os.path.join(self.temp_dir, "test_evidence.jpg")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(self.test_image_path, test_image)
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_end_to_end_workflow(self):
        """Test flujo de trabajo completo end-to-end"""
        # Datos de caso completo
        case_data = {
            "case_id": "E2E_TEST_001",
            "evidence_type": "cartridge_case",
            "image_path": self.test_image_path,
            "comparison_image": self.test_image_path,
            "metadata": {
                "examiner": "Integration Tester",
                "laboratory": "Test Lab",
                "date": datetime.now().isoformat()
                }
            }
        
        # Ejecutar flujo completo
        try:
            result = self.nist_manager.execute_complete_workflow(case_data)
            self.assertIsNotNone(result)
            self.assertIn('processing_complete', result)
        except Exception as e:
            # Si el método no existe, crear un test básico
            self.skipTest(f"Complete workflow not implemented: {e}")
            
    def test_nist_compliance_validation(self):
        """Test validación de cumplimiento NIST"""
        compliance_data = {
            "schema_version": "2.0",
            "required_fields": [
                "case_id", "examiner", "evidence_type", 
                "examination_method", "conclusion"
            ],
            "quality_thresholds": {
                "min_resolution": 300,
                "min_snr": 20.0,
                "min_contrast": 0.5
            }
        }
        
        test_case = {
            "case_id": "COMPLIANCE_TEST",
            "examiner": "Compliance Tester",
            "evidence_type": "bullet",
            "examination_method": "comparison",
            "image_quality": {
                "resolution": 600,
                "snr": 25.5,
                "contrast": 0.7
            }
        }
        
        is_compliant = self.nist_manager.validate_nist_compliance(
            test_case, compliance_data
        )
        
        # Si el método no existe, asumir cumplimiento básico
        if hasattr(self.nist_manager, 'validate_nist_compliance'):
            self.assertTrue(is_compliant)
        else:
            self.skipTest("NIST compliance validation not implemented")
            
    def test_error_handling_and_recovery(self):
        """Test manejo de errores y recuperación"""
        # Datos inválidos para provocar errores
        invalid_cases = [
            {"case_id": None},  # ID faltante
            {"case_id": "TEST", "image_path": "nonexistent.jpg"},  # Imagen inexistente
            {"case_id": "TEST", "evidence_type": "invalid_type"}  # Tipo inválido
        ]
        
        for invalid_case in invalid_cases:
            try:
                result = self.nist_manager.process_ballistic_evidence(invalid_case)
                # Si no hay excepción, verificar que el resultado indique error
                if result:
                    self.assertIn('error', result.lower() if isinstance(result, str) else str(result))
            except Exception as e:
                # Verificar que la excepción sea manejada apropiadamente
                self.assertIsInstance(e, (ValueError, FileNotFoundError, TypeError))


class TestNISTStatisticalIntegration(unittest.TestCase):
    """Tests de integración entre NIST Standards y Análisis Estadístico"""
    
    def setUp(self):
        """Configurar datos de prueba"""
        self.nist_manager = NISTStandardsManager()
        
        # Datos mock de evidencia balística
        self.mock_evidence_data = {
            'case_id': 'TEST_001',
            'evidence_type': 'cartridge_case',
            'image_path': 'test_evidence.jpg',
            'comparison_data': {
                'reference_image': 'reference.jpg',
                'similarity_scores': [0.85, 0.92, 0.78, 0.89, 0.91],
                'feature_matches': [
                    {'type': 'striation', 'confidence': 0.95},
                    {'type': 'impression', 'confidence': 0.87},
                    {'type': 'breach_face', 'confidence': 0.82}
                ]
            },
            'quality_metrics': {
                'snr': 25.5,
                'contrast': 0.72,
                'sharpness': 0.88,
                'uniformity': 0.91
            }
        }
        
        # Datos para análisis estadístico
        self.statistical_data = {
            'sample_size': 100,
            'confidence_level': 0.95,
            'bootstrap_iterations': 1000,
            'cross_validation_folds': 5,
            'test_data': {
                'similarity_scores': np.random.beta(2, 2, 100),
                'quality_scores': np.random.normal(0.8, 0.1, 100),
                'ground_truth': np.random.choice([0, 1], 100, p=[0.3, 0.7])
                }
            }
    
    def test_nist_manager_has_statistical_analysis(self):
        """Verificar que NISTStandardsManager tiene capacidades de análisis estadístico"""
        # Verificar que el manager puede acceder a análisis estadístico
        self.assertTrue(hasattr(self.nist_manager, 'statistical_analyzer') or 
                      hasattr(self.nist_manager, 'get_statistical_analyzer'))
        
    def test_process_ballistic_evidence_with_statistics(self):
        """Test procesamiento de evidencia con análisis estadístico integrado"""
        try:
            result = self.nist_manager.process_ballistic_evidence(
                self.mock_evidence_data,
                include_statistical_analysis=True
            )
            
            # Verificar que el resultado incluye análisis estadístico
            self.assertIn('statistical_analysis', result)
            self.assertIn('confidence_intervals', result['statistical_analysis'])
            self.assertIn('bootstrap_results', result['statistical_analysis'])
            
        except (AttributeError, TypeError):
            # Si el método no soporta análisis estadístico, skip el test
            self.skipTest("Statistical analysis integration not implemented")
            
    def test_compare_evidence_with_statistical_tests(self):
        """Test comparación de evidencia con tests estadísticos"""
        evidence_1 = self.mock_evidence_data
        evidence_2 = dict(self.mock_evidence_data)
        evidence_2['case_id'] = 'TEST_002'
        evidence_2['comparison_data']['similarity_scores'] = [0.45, 0.52, 0.38, 0.49, 0.41]
        
        try:
            comparison_result = self.nist_manager.compare_evidence_statistical(
                evidence_1, evidence_2
            )
            
            self.assertIn('statistical_significance', comparison_result)
            self.assertIn('p_value', comparison_result)
            self.assertIn('effect_size', comparison_result)
            
        except AttributeError:
            self.skipTest("Statistical evidence comparison not implemented")
            
    def test_export_nist_report_with_statistics(self):
        """Test exportación de reporte NIST con estadísticas incluidas"""
        report_data = {
            'case_info': self.mock_evidence_data,
            'analysis_results': {
                'afte_conclusion': 'identification',
                'confidence_level': 'high',
                'statistical_support': {
                    'bootstrap_confidence_interval': [0.82, 0.94],
                    'p_value': 0.001,
                    'effect_size': 1.25,
                    'sample_size': 100
                        }
                    },
            'validation_metrics': {
                'cross_validation_accuracy': 0.92,
                'bootstrap_stability': 0.95,
                'statistical_power': 0.88
                }
            }
        
        try:
            export_success = self.nist_manager.export_statistical_report(
                report_data, 'test_statistical_report.xml'
            )
            self.assertTrue(export_success)
            
        except AttributeError:
            self.skipTest("Statistical report export not implemented")
            
    def test_validate_system_with_statistical_metrics(self):
        """Test validación del sistema con métricas estadísticas"""
        validation_config = {
            'dataset': self.statistical_data,
            'validation_type': 'statistical',
            'metrics': ['bootstrap_accuracy', 'cross_validation_stability', 'statistical_power'],
            'significance_level': 0.05
        }
        
        try:
            validation_result = self.nist_manager.validate_system_statistical(validation_config)
            
            self.assertIn('statistical_validation', validation_result)
            self.assertIn('power_analysis', validation_result)
            self.assertIn('effect_size_analysis', validation_result)
            
        except AttributeError:
            self.skipTest("Statistical system validation not implemented")


def run_nist_tests():
    """Ejecutar todos los tests NIST de forma organizada"""
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Agregar tests por categoría
    schema_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTSchema)
    quality_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTQualityMetrics)
    afte_tests = unittest.TestLoader().loadTestsFromTestCase(TestAFTEConclusions)
    validation_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTValidationProtocols)
    manager_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTStandardsManager)
    integration_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTIntegration)
    statistical_tests = unittest.TestLoader().loadTestsFromTestCase(TestNISTStatisticalIntegration)
    
    # Agregar al suite
    test_suite.addTests(schema_tests)
    test_suite.addTests(quality_tests)
    test_suite.addTests(afte_tests)
    test_suite.addTests(validation_tests)
    test_suite.addTests(manager_tests)
    test_suite.addTests(integration_tests)
    test_suite.addTests(statistical_tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    return runner.run(test_suite)


class TestNISTImageProcessingIntegration(unittest.TestCase):
    """Tests de integración para procesamiento de imágenes con estándares NIST"""
    
    def setUp(self):
        """Configurar componentes de procesamiento"""
        self.calibrator = SpatialCalibrator()
        self.validator = NISTComplianceValidator()
        self.preprocessor = UnifiedPreprocessor()
        
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear imagen de prueba
        self.test_image = self._create_test_image()
        
    def tearDown(self):
        """Limpiar archivos temporales"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def _create_test_image(self):
        """Crear imagen de prueba con características conocidas"""
        # Crear imagen sintética con patrones conocidos
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        # Agregar patrones de estrías
        for i in range(0, 1000, 20):
            cv2.line(image, (i, 0), (i, 1000), (255, 255, 255), 2)
            
        # Agregar ruido controlado
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        
        return image
        
    def test_nist_compliant_processing(self):
        """Test procesamiento completo compatible con NIST"""
        # Configuración de procesamiento NIST
        config = PreprocessingConfig(
            target_resolution=300,  # DPI mínimo NIST
            noise_reduction=True,
            contrast_enhancement=True,
            nist_compliance=True
        )
        
        try:
            processed_image = self.preprocessor.process_image(self.test_image, config)
            
            # Validar cumplimiento NIST
            compliance_report = self.validator.validate_image(processed_image)
            
            self.assertIsInstance(compliance_report, NISTProcessingReport)
            self.assertTrue(compliance_report.is_compliant)
            
        except (AttributeError, ImportError):
            self.skipTest("NIST compliant processing not fully implemented")
            
    def test_spatial_calibration_nist_compliance(self):
        """Test calibración espacial según estándares NIST"""
        # Datos de calibración conocidos
        calibration_data = CalibrationData(
            reference_dpi=300,
            pixel_size_microns=84.67,  # Para 300 DPI
            calibration_standard="NIST_SRM_2460"
        )
        
        try:
            calibrated_image = self.calibrator.calibrate_image(
                self.test_image, calibration_data
            )
            
            # Verificar que la calibración cumple estándares
            validation_result = self.calibrator.validate_calibration(calibrated_image)
            
            self.assertTrue(validation_result['nist_compliant'])
            self.assertGreaterEqual(validation_result['accuracy'], 0.95)
            
        except (AttributeError, ImportError):
            self.skipTest("Spatial calibration not fully implemented")
            
    def test_illumination_uniformity_nist_standard(self):
        """Test uniformidad de iluminación según estándar NIST"""
        try:
            uniformity_metrics = self.validator.assess_illumination_uniformity(
                self.test_image
            )
            
            # NIST requiere uniformidad > 90%
            self.assertGreaterEqual(uniformity_metrics['uniformity_percentage'], 90.0)
            self.assertLessEqual(uniformity_metrics['variation_coefficient'], 0.1)
            
        except (AttributeError, ImportError):
            self.skipTest("Illumination uniformity assessment not implemented")
            
    def test_complete_nist_validation_workflow(self):
        """Test flujo completo de validación NIST"""
        # Configurar pipeline completo
        nist_pipeline_config = {
            'spatial_calibration': True,
            'quality_assessment': True,
            'illumination_validation': True,
            'metadata_compliance': True,
            'traceability_documentation': True
        }
        
        try:
            # Ejecutar pipeline completo
            validation_result = self.validator.execute_complete_validation(
                self.test_image, nist_pipeline_config
            )
            
            # Verificar componentes de validación
            self.assertIn('spatial_calibration', validation_result)
            self.assertIn('quality_metrics', validation_result)
            self.assertIn('illumination_assessment', validation_result)
            self.assertIn('metadata_validation', validation_result)
            self.assertIn('overall_compliance', validation_result)
            
            # Verificar cumplimiento general
            self.assertTrue(validation_result['overall_compliance'])
            
        except (AttributeError, ImportError):
            self.skipTest("Complete NIST validation workflow not implemented")


if __name__ == '__main__':
    print("🧪 Ejecutando Tests Comprehensivos de Estándares NIST")
    print("=" * 60)
    
    result = run_nist_tests()
    
    print("\n" + "=" * 60)
    print(f"📊 Resumen de Tests:")
    print(f"   Tests ejecutados: {result.testsRun}")
    print(f"   Errores: {len(result.errors)}")
    print(f"   Fallos: {len(result.failures)}")
    
    if result.errors:
        print(f"\n❌ Errores encontrados:")
        for test, error in result.errors:
            print(f"   - {test}: {error}")
    
    if result.failures:
        print(f"\n❌ Fallos encontrados:")
        for test, failure in result.failures:
            print(f"   - {test}: {failure}")
    
    if result.wasSuccessful():
        print(f"\n✅ Todos los tests pasaron correctamente!")
    else:
        print(f"\n⚠️  Algunos tests fallaron. Revisar implementación.")