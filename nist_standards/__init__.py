"""Módulo de Estándares NIST para Sistemas Balísticos Forenses
==========================================================

Este módulo implementa los estándares NIST para análisis balístico forense,
incluyendo formatos de datos, métricas de calidad y protocolos de validación.

Componentes principales:
- Schema XML NIST para intercambio de datos
- Métricas de calidad de imagen según NIST
- Categorías AFTE para conclusiones forenses
- Protocolos de validación y verificación
"""

from .nist_schema import (
    NISTSchema,
    NISTDataExporter,
    NISTDataImporter,
    NISTMetadata,
    NISTImageData,
    NISTFeatureData,
    NISTComparisonResult,
    EvidenceType,
    ExaminationMethod
)

from .quality_metrics import (
    NISTQualityMetrics,
    NISTQualityReport
)

from .afte_conclusions import (
    AFTEConclusionEngine,
    AFTEConclusion,
    AFTEAnalysisResult,
    ConfidenceLevel,
    FeatureType,
    FeatureMatch
)

from .validation_protocols import (
    NISTValidationProtocols,
    ValidationResult,
    ValidationLevel,
    ValidationDataset,
    MetricType
)

from .statistical_analysis import (
    AdvancedStatisticalAnalysis, 
    StatisticalTest, 
    CorrectionMethod,
    BootstrapResult,
    StatisticalTestResult,
    MultipleComparisonResult
)

__version__ = "1.0.0"
__author__ = "SEACABA Team"
__description__ = "Implementación de Estándares NIST para Análisis Balístico Forense"

# Configuración por defecto
DEFAULT_NIST_CONFIG = {
    'xml_schema_version': '1.0',
    'quality_thresholds': {
        'snr_min': 20.0,
        'contrast_min': 0.3,
        'uniformity_min': 0.8
    },
    'afte_confidence_threshold': 0.85,
    'validation_k_folds': 10
}


class NISTStandardsManager:
    """
    Gestor principal de estándares NIST que integra todos los componentes
    """
    
    def __init__(self, config: dict = None):
        """
        Inicializa el gestor de estándares NIST
        
        Args:
            config: Configuración personalizada (opcional)
        """
        self.config = {**DEFAULT_NIST_CONFIG, **(config or {})}
        
        # Inicializar componentes
        self.schema = NISTSchema()
        self.data_exporter = NISTDataExporter()
        self.data_importer = NISTDataImporter()
        self.quality_metrics = NISTQualityMetrics()
        self.afte_engine = AFTEConclusionEngine()
        self.validation_protocols = NISTValidationProtocols()
        self.statistical_analysis = AdvancedStatisticalAnalysis()
    
    def process_ballistic_evidence(self, image_path: str, metadata: dict) -> dict:
        """
        Procesa evidencia balística completa según estándares NIST
        
        Args:
            image_path: Ruta de la imagen
            metadata: Metadatos de la evidencia
            
        Returns:
            dict: Resultado completo del procesamiento NIST
        """
        try:
            import cv2
            import numpy as np
            from datetime import datetime
            
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # 1. Análisis de calidad NIST
            quality_report = self.quality_metrics.analyze_image_quality(image)
            
            # 2. Crear metadatos NIST
            nist_metadata = NISTMetadata(
                case_id=metadata.get('case_id', 'unknown'),
                evidence_id=metadata.get('evidence_id', 'unknown'),
                examiner_id=metadata.get('examiner_id', 'unknown'),
                examination_date=datetime.now(),
                evidence_type=EvidenceType(metadata.get('evidence_type', 'cartridge_case')),
                examination_method=ExaminationMethod(metadata.get('examination_method', 'digital_microscopy')),
                laboratory_id=metadata.get('laboratory_id', 'unknown'),
                instrument_id=metadata.get('instrument_id', 'unknown')
            )
            
            # 3. Crear datos de imagen NIST
            nist_image_data = NISTImageData(
                image_id=f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=image_path,
                image_format=image_path.split('.')[-1].upper(),
                width=image.shape[1],
                height=image.shape[0],
                bit_depth=8,
                color_space="BGR",
                resolution_dpi=metadata.get('resolution_dpi', 300),
                acquisition_parameters=metadata.get('acquisition_parameters', {}),
                quality_metrics=quality_report.to_dict()
            )
            
            # 4. Resultado integrado
            result = {
                'nist_metadata': nist_metadata,
                'image_data': nist_image_data,
                'quality_report': quality_report,
                'processing_timestamp': datetime.now().isoformat(),
                'nist_compliance': quality_report.overall_quality >= 0.7,
                'recommendations': quality_report.recommendations
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"Error procesando evidencia: {e}",
                'nist_compliance': False,
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def compare_evidence_nist(self, evidence1_data: dict, evidence2_data: dict, 
                             comparison_features: dict) -> dict:
        """
        Compara dos evidencias según estándares NIST y AFTE
        
        Args:
            evidence1_data: Datos de la primera evidencia
            evidence2_data: Datos de la segunda evidencia
            comparison_features: Características extraídas para comparación
            
        Returns:
            dict: Resultado de comparación NIST/AFTE
        """
        try:
            from datetime import datetime
            
            # Crear datos de características NIST
            features1 = NISTFeatureData(
                feature_id=f"feat1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                extraction_method="ORB",
                feature_vector=comparison_features.get('features1', []),
                keypoints_count=len(comparison_features.get('keypoints1', [])),
                descriptors_count=len(comparison_features.get('descriptors1', [])),
                extraction_parameters=comparison_features.get('extraction_params', {}),
                quality_score=evidence1_data.get('quality_report', {}).get('overall_quality', 0.0)
            )
            
            features2 = NISTFeatureData(
                feature_id=f"feat2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                extraction_method="ORB",
                feature_vector=comparison_features.get('features2', []),
                keypoints_count=len(comparison_features.get('keypoints2', [])),
                descriptors_count=len(comparison_features.get('descriptors2', [])),
                extraction_parameters=comparison_features.get('extraction_params', {}),
                quality_score=evidence2_data.get('quality_report', {}).get('overall_quality', 0.0)
            )
            
            # Análisis AFTE
            afte_result = self.afte_engine.analyze_comparison(
                features1.to_dict(),
                features2.to_dict(),
                comparison_features.get('matches', []),
                comparison_features.get('similarity_score', 0.0)
            )
            
            # Crear resultado de comparación NIST
            nist_comparison = NISTComparisonResult(
                comparison_id=f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                evidence1_id=evidence1_data.get('nist_metadata').evidence_id,
                evidence2_id=evidence2_data.get('nist_metadata').evidence_id,
                comparison_method="feature_matching",
                similarity_score=comparison_features.get('similarity_score', 0.0),
                match_count=len(comparison_features.get('matches', [])),
                comparison_parameters=comparison_features.get('comparison_params', {}),
                statistical_measures=afte_result.statistical_analysis,
                examiner_conclusion=afte_result.conclusion.value,
                confidence_level=afte_result.confidence_level.value,
                comparison_notes=afte_result.analysis_notes
            )
            
            # Resultado integrado
            result = {
                'nist_comparison': nist_comparison,
                'afte_analysis': afte_result,
                'features1': features1,
                'features2': features2,
                'comparison_timestamp': datetime.now().isoformat(),
                'nist_compliance': True,
                'quality_assessment': {
                    'evidence1_quality': evidence1_data.get('quality_report', {}).get('overall_quality', 0.0),
                    'evidence2_quality': evidence2_data.get('quality_report', {}).get('overall_quality', 0.0),
                    'comparison_reliability': afte_result.reliability_score
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"Error en comparación NIST: {e}",
                'nist_compliance': False,
                'comparison_timestamp': datetime.now().isoformat()
            }
    
    def export_nist_report(self, analysis_data: dict, output_path: str, 
                          format_type: str = 'xml') -> bool:
        """
        Exporta reporte completo en formato NIST
        
        Args:
            analysis_data: Datos del análisis
            output_path: Ruta de salida
            format_type: Tipo de formato ('xml' o 'json')
            
        Returns:
            bool: True si se exportó correctamente
        """
        try:
            if format_type.lower() == 'xml':
                return self.data_exporter.export_to_xml(analysis_data, output_path)
            elif format_type.lower() == 'json':
                return self.data_exporter.export_to_json(analysis_data, output_path)
            else:
                raise ValueError(f"Formato no soportado: {format_type}")
                
        except Exception as e:
            print(f"Error exportando reporte NIST: {e}")
            return False
    
    def validate_system_nist(self, validation_dataset, classifier_func, 
                           validation_level: ValidationLevel = ValidationLevel.INTERNAL) -> ValidationResult:
        """
        Valida el sistema completo según protocolos NIST
        
        Args:
            validation_dataset: Dataset de validación
            classifier_func: Función clasificadora
            validation_level: Nivel de validación
            
        Returns:
            ValidationResult: Resultado de validación
        """
        return self.validation_protocols.perform_k_fold_validation(
            validation_dataset, classifier_func, 
            k_folds=self.config['validation_k_folds'],
            validation_level=validation_level
        )


__all__ = [
    # Schema XML
    'NISTSchema',
    'NISTDataExporter', 
    'NISTDataImporter',
    'NISTMetadata',
    'NISTImageData',
    'NISTFeatureData',
    'NISTComparisonResult',
    'EvidenceType',
    'ExaminationMethod',
    
    # Métricas de calidad
    'NISTQualityMetrics',
    'NISTQualityReport',
    
    # Conclusiones AFTE
    'AFTEConclusionEngine',
    'AFTEConclusion',
    'AFTEAnalysisResult',
    'ConfidenceLevel',
    'FeatureType',
    'FeatureMatch',
    
    # Protocolos de validación
    'NISTValidationProtocols',
    'ValidationResult',
    'ValidationLevel',
    'ValidationDataset',
    'MetricType',
    
    # Gestor principal
    'NISTStandardsManager',
    
    # Configuración
    'DEFAULT_NIST_CONFIG'
]