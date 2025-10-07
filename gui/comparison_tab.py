#!/usr/bin/env python3
"""
Pestaña de Análisis Comparativo Balístico
Sistema SIGeC-Balistica- Análisis de Cartuchos y Balas Automático

Dos modos especializados:
1. Comparación Directa (Evidencia A vs Evidencia B) con análisis CMC
2. Búsqueda en Base de Datos con ranking y visualización de coincidencias
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QSplitter, QFrame, QSpacerItem,
    QSizePolicy, QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QListWidget, QListWidgetItem, QSlider, QDoubleSpinBox, QButtonGroup,
    QRadioButton, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QPen, QColor

from .shared_widgets import (
    ImageDropZone, ResultCard, CollapsiblePanel, StepIndicator, 
    ProgressCard, ImageViewer
)
from .model_selector_dialog import ModelSelectorDialog
from .synchronized_viewer import SynchronizedViewer
from .interactive_matching_widget import InteractiveMatchingWidget
from .dynamic_results_panel import DynamicResultsPanel
from .assisted_alignment import AssistedAlignmentWidget
from .correlation_heatmap import CorrelationHeatmapWidget
from .interactive_cmc_widget import InteractiveCMCWidget
from .gallery_search_widget import GallerySearchWidget

# Importaciones para validación NIST
try:
    from image_processing.nist_compliance_validator import NISTComplianceValidator, NISTProcessingReport
    from nist_standards.quality_metrics import NISTQualityMetrics, NISTQualityReport
    from nist_standards.afte_conclusions import AFTEConclusionEngine, AFTEConclusion
    from nist_standards.validation_protocols import NISTValidationProtocols
    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False

# Importaciones para Deep Learning
try:
    from deep_learning.ballistic_dl_models import BallisticCNN
    from deep_learning.models.siamese_models import SiameseNetwork
    from deep_learning.config.unified_config import ModelConfig, ModelType
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class BallisticComparisonWorker(QThread):
    """Worker thread especializado para comparaciones balísticas con pipeline unificado"""
    
    progressUpdated = pyqtSignal(int, str)
    comparisonCompleted = pyqtSignal(dict)
    comparisonError = pyqtSignal(str)
    
    def __init__(self, comparison_params: dict):
        super().__init__()
        self.comparison_params = comparison_params
        self.should_stop = False
        
        # Configurar logger
        import logging
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Ejecuta la comparación balística en segundo plano"""
        try:
            mode = self.comparison_params.get('mode', 'direct')
            
            if mode == 'direct':
                self.run_unified_ballistic_comparison()
            else:
                self.run_ballistic_database_search()
                
        except Exception as e:
            self.comparisonError.emit(str(e))
            
    def run_unified_ballistic_comparison(self):
        """Ejecuta comparación directa usando el pipeline científico unificado"""
        try:
            # Importar pipeline unificado y componentes
            from core.unified_pipeline import ScientificPipeline, PipelineLevel
            from core.pipeline_config import create_pipeline_config
            from matching.unified_matcher import AlgorithmType
            
            # Obtener parámetros de comparación
            img1_path = self.comparison_params.get('evidence_a')
            img2_path = self.comparison_params.get('evidence_b')
            evidence_type = self.comparison_params.get('evidence_type', 'cartridge_case')
            algorithm_name = self.comparison_params.get('algorithm', 'ORB')
            use_deep_learning = self.comparison_params.get('use_deep_learning', False)
            enable_nist_validation = self.comparison_params.get('enable_nist_validation', True)
            
            if not img1_path or not img2_path:
                self.comparisonError.emit("Rutas de imágenes no válidas")
                return
            
            self.progressUpdated.emit(10, "Inicializando pipeline científico unificado...")
            
            # Determinar nivel de análisis basado en configuración
            analysis_level = PipelineLevel.STANDARD
            if use_deep_learning:
                analysis_level = PipelineLevel.ADVANCED
            if enable_nist_validation:
                analysis_level = PipelineLevel.FORENSIC
            
            # Crear configuración del pipeline
            try:
                pipeline_config = create_pipeline_config(analysis_level)
                # Configurar algoritmo específico
                if hasattr(pipeline_config, 'matching'):
                    pipeline_config.matching.algorithm = getattr(AlgorithmType, algorithm_name, AlgorithmType.ORB)
            except:
                # Fallback si no está disponible la configuración
                pipeline_config = None
            
            self.progressUpdated.emit(20, "Inicializando componentes del sistema...")
            
            # Inicializar pipeline unificado
            pipeline = ScientificPipeline(pipeline_config)
            
            self.progressUpdated.emit(30, "Ejecutando análisis completo...")
            
            # Ejecutar pipeline completo
            pipeline_result = pipeline.process_comparison(img1_path, img2_path)
            
            self.progressUpdated.emit(80, "Procesando resultados...")
            
            # Formatear resultados para la interfaz
            results = self._format_unified_pipeline_results(
                pipeline_result, evidence_type, img1_path, img2_path
            )
            
            # Agregar análisis de deep learning si está habilitado
            if use_deep_learning and DEEP_LEARNING_AVAILABLE:
                self.progressUpdated.emit(90, "Ejecutando análisis de deep learning...")
                dl_results = self._run_deep_learning_analysis(img1_path, img2_path)
                results['deep_learning'] = dl_results
            
            self.progressUpdated.emit(100, "Análisis completado")
            self.comparisonCompleted.emit(results)
            
        except Exception as e:
            self.logger.error(f"Error en comparación unificada: {e}")
            self.comparisonError.emit(f"Error en análisis: {str(e)}")
    
    def _format_unified_pipeline_results(self, pipeline_result, evidence_type, img1_path, img2_path):
        """Formatea los resultados del pipeline unificado para la interfaz"""
        try:
            # Información básica
            results = {
                'comparison_type': 'direct_unified',
                'evidence_type': evidence_type,
                'evidence_a': img1_path,
                'evidence_b': img2_path,
                'timestamp': pipeline_result.analysis_timestamp,
                'processing_time': pipeline_result.processing_time,
                'success': pipeline_result.afte_conclusion != pipeline_result.afte_conclusion.UNSUITABLE
            }
            
            # Resultados de calidad NIST
            if pipeline_result.image1_quality and pipeline_result.image2_quality:
                results['quality_assessment'] = {
                    'image1_quality': {
                        'score': pipeline_result.image1_quality.quality_score,
                        'level': pipeline_result.image1_quality.quality_level.value if hasattr(pipeline_result.image1_quality, 'quality_level') else 'unknown',
                        'metrics': pipeline_result.image1_quality.metrics if hasattr(pipeline_result.image1_quality, 'metrics') else {}
                    },
                    'image2_quality': {
                        'score': pipeline_result.image2_quality.quality_score,
                        'level': pipeline_result.image2_quality.quality_level.value if hasattr(pipeline_result.image2_quality, 'quality_level') else 'unknown',
                        'metrics': pipeline_result.image2_quality.metrics if hasattr(pipeline_result.image2_quality, 'metrics') else {}
                    },
                    'assessment_passed': pipeline_result.quality_assessment_passed
                }
            
            # Resultados de matching
            if pipeline_result.match_result:
                results['matching'] = {
                    'algorithm': pipeline_result.match_result.algorithm.value if hasattr(pipeline_result.match_result, 'algorithm') else 'unknown',
                    'similarity_score': pipeline_result.similarity_score,
                    'quality_weighted_score': pipeline_result.quality_weighted_score,
                    'keypoints_1': pipeline_result.match_result.keypoints_1 if hasattr(pipeline_result.match_result, 'keypoints_1') else 0,
                    'keypoints_2': pipeline_result.match_result.keypoints_2 if hasattr(pipeline_result.match_result, 'keypoints_2') else 0,
                    'matches_found': pipeline_result.match_result.matches_found if hasattr(pipeline_result.match_result, 'matches_found') else 0,
                    'good_matches': pipeline_result.match_result.good_matches if hasattr(pipeline_result.match_result, 'good_matches') else 0
                }
            
            # Resultados CMC
            if pipeline_result.cmc_result:
                results['cmc_analysis'] = {
                    'cmc_count': pipeline_result.cmc_count,
                    'cmc_passed': pipeline_result.cmc_passed,
                    'cmc_threshold': pipeline_result.cmc_result.cmc_threshold if hasattr(pipeline_result.cmc_result, 'cmc_threshold') else 0,
                    'correlation_cells': pipeline_result.cmc_result.correlation_cells if hasattr(pipeline_result.cmc_result, 'correlation_cells') else []
                }
            
            # Conclusión AFTE
            results['afte_conclusion'] = {
                'conclusion': pipeline_result.afte_conclusion.value,
                'confidence': pipeline_result.confidence,
                'reasoning': self._get_afte_reasoning(pipeline_result)
            }
            
            # ROI detectadas
            if pipeline_result.roi1_detected or pipeline_result.roi2_detected:
                results['roi_detection'] = {
                    'image1_roi_detected': pipeline_result.roi1_detected,
                    'image2_roi_detected': pipeline_result.roi2_detected,
                    'image1_regions': pipeline_result.roi1_regions,
                    'image2_regions': pipeline_result.roi2_regions
                }
            
            # Pasos de preprocesamiento aplicados
            if pipeline_result.preprocessing_steps:
                results['preprocessing'] = {
                    'successful': pipeline_result.preprocessing_successful,
                    'steps_applied': pipeline_result.preprocessing_steps
                }
            
            # Mensajes de error y advertencias
            if pipeline_result.error_messages:
                results['errors'] = pipeline_result.error_messages
            if pipeline_result.warnings:
                results['warnings'] = pipeline_result.warnings
            
            # Datos intermedios para visualización
            if pipeline_result.intermediate_results:
                results['intermediate_data'] = pipeline_result.intermediate_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error formateando resultados del pipeline: {e}")
            return {
                'comparison_type': 'direct_unified',
                'success': False,
                'error': f"Error formateando resultados: {str(e)}"
            }
    
    def _get_afte_reasoning(self, pipeline_result):
        """Genera explicación del razonamiento para la conclusión AFTE"""
        reasoning = []
        
        if pipeline_result.quality_assessment_passed:
            reasoning.append("Imágenes cumplen estándares de calidad NIST")
        else:
            reasoning.append("Calidad de imagen insuficiente según estándares NIST")
        
        if pipeline_result.match_result:
            score = pipeline_result.similarity_score
            if score > 0.8:
                reasoning.append(f"Alta similitud detectada (score: {score:.3f})")
            elif score > 0.5:
                reasoning.append(f"Similitud moderada detectada (score: {score:.3f})")
            else:
                reasoning.append(f"Baja similitud detectada (score: {score:.3f})")
        
        if pipeline_result.cmc_passed:
            reasoning.append(f"Análisis CMC exitoso ({pipeline_result.cmc_count} correlaciones)")
        elif pipeline_result.cmc_result:
            reasoning.append(f"Análisis CMC insuficiente ({pipeline_result.cmc_count} correlaciones)")
        
        return "; ".join(reasoning) if reasoning else "Análisis completado"
    
    def _run_deep_learning_analysis(self, img1_path, img2_path):
        """Ejecuta análisis adicional usando modelos de deep learning"""
        try:
            from deep_learning.ballistic_dl_models import CNNFeatureExtractor, SiameseNetwork
            import cv2
            
            # Cargar imágenes
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return {'error': 'No se pudieron cargar las imágenes para análisis DL'}
            
            results = {}
            
            # Análisis con CNN Feature Extractor
            try:
                cnn_extractor = CNNFeatureExtractor()
                features1 = cnn_extractor.extract_features(img1)
                features2 = cnn_extractor.extract_features(img2)
                
                # Calcular similitud usando características CNN
                if features1 is not None and features2 is not None:
                    similarity = np.dot(features1.flatten(), features2.flatten()) / (
                        np.linalg.norm(features1) * np.linalg.norm(features2)
                    )
                    results['cnn_similarity'] = float(similarity)
                
            except Exception as e:
                results['cnn_error'] = str(e)
            
            # Análisis con Siamese Network (si está disponible)
            try:
                siamese_net = SiameseNetwork()
                similarity_score = siamese_net.predict_similarity(img1, img2)
                results['siamese_similarity'] = float(similarity_score)
                
            except Exception as e:
                results['siamese_error'] = str(e)
            
            return results
            
        except Exception as e:
            return {'error': f'Error en análisis de deep learning: {str(e)}'}

    def run_direct_ballistic_comparison(self):
        """Método legacy mantenido para compatibilidad"""
        # Redirigir al nuevo método unificado
        return self.run_unified_ballistic_comparison()
    
    def _format_comparison_results(self, match_result, evidence_type):
        """Formatea los resultados del UnifiedMatcher para la UI"""
        # Determinar conclusión AFTE basada en similarity score
        similarity = match_result.similarity_score / 100.0  # Normalizar a 0-1
        
        if similarity >= 0.85:
            afte_conclusion = "Identification"
            result_type = "success"
        elif similarity >= 0.70:
            afte_conclusion = "Inconclusive"
            result_type = "warning"
        else:
            afte_conclusion = "Elimination"
            result_type = "error"
        
        # Extraer datos adicionales del match_result
        match_data = match_result.match_data or {}
        
        return {
            'mode': 'direct',
            'evidence_type': evidence_type,
            'image_a': self.comparison_params.get('evidence_a'),
            'image_b': self.comparison_params.get('evidence_b'),
            'algorithm': match_result.algorithm,
            'similarity_score': match_result.similarity_score,
            'confidence': match_result.confidence,
            'cmc_score': similarity,  # Para compatibilidad con UI existente
            'afte_conclusion': afte_conclusion,
            'result_type': result_type,
            'ballistic_features': {
                'total_keypoints_a': match_result.total_keypoints1,
                'total_keypoints_b': match_result.total_keypoints2,
                'total_matches': match_result.total_matches,
                'good_matches': match_result.good_matches,
                'geometric_consistency': match_result.geometric_consistency,
                'match_quality': match_result.match_quality
            },
            'cmc_analysis': {
                'total_cells': match_data.get('total_cells', 0),
                'valid_cells': match_data.get('valid_cells', 0),
                'congruent_cells': match_data.get('congruent_cells', 0),
                'convergence_score': match_result.confidence,
                'cell_correlation_threshold': 0.6
            },
            'statistical_analysis': {
                'confidence_interval_lower': match_result.confidence_interval_lower,
                'confidence_interval_upper': match_result.confidence_interval_upper,
                'bootstrap_confidence_level': match_result.bootstrap_confidence_level,
                'bootstrap_std_error': match_result.bootstrap_std_error
            },
            'visualizations': {
                'algorithm_used': match_result.algorithm,
                'processing_time': match_result.processing_time,
                'keypoint_density': match_result.keypoint_density
            },
            'quality_metrics': {
                'image_a_quality': match_result.image1_quality_score,
                'image_b_quality': match_result.image2_quality_score,
                'combined_quality': match_result.combined_quality_score,
                'quality_weighted_similarity': match_result.quality_weighted_similarity
            }
        }
        
    def run_ballistic_database_search(self):
        """Ejecuta búsqueda balística real en base de datos"""
        try:
            # Obtener parámetros de búsqueda
            query_path = self.comparison_params.get('query_image', '')
            max_results = self.comparison_params.get('max_results', 10)
            similarity_threshold = self.comparison_params.get('similarity_threshold', 0.3)
            evidence_type = self.comparison_params.get('evidence_type', 'cartridge_case')
            
            if not query_path or not os.path.exists(query_path):
                raise ValueError("Imagen de consulta no válida o no encontrada")
            
            # Paso 1: Inicializar componentes
            self.progressUpdated.emit(10, "Inicializando matcher y base de datos...")
            
            from matching.unified_matcher import UnifiedMatcher, MatchingConfig, AlgorithmType
            from database.vector_db import VectorDatabase
            import cv2
            
            # Configurar matcher con parámetros optimizados
            config = MatchingConfig(
                algorithm=AlgorithmType.ORB,  # Usar ORB por defecto
                max_features=5000,
                distance_threshold=0.75,
                min_matches=10
            )
            matcher = UnifiedMatcher(config)
            db = VectorDatabase()
            
            # Paso 2: Extraer características de consulta
            self.progressUpdated.emit(20, "Extrayendo características de imagen de consulta...")
            
            # Cargar y procesar imagen de consulta
            query_image = cv2.imread(query_path)
            if query_image is None:
                raise ValueError("No se pudo cargar la imagen de consulta")
            
            # Extraer características usando el matcher
            query_features = matcher.extract_features(query_image)
            if query_features is None or len(query_features) == 0:
                raise ValueError("No se pudieron extraer características de la imagen de consulta")
            
            # Paso 3: Buscar en base de datos vectorial
            self.progressUpdated.emit(40, "Buscando coincidencias en base de datos FAISS...")
            
            # Convertir características a vector para búsqueda FAISS
            query_vector = self._features_to_vector(query_features)
            
            # Buscar vectores similares
            similar_vectors = db.search_similar_vectors(
                query_vector, 
                k=max_results * 2,  # Buscar más para filtrar después
                distance_threshold=1.0 - similarity_threshold  # Convertir similitud a distancia
            )
            
            if not similar_vectors:
                # Si no hay resultados en FAISS, devolver resultado vacío
                results = self._format_empty_search_results(query_path, evidence_type)
                self.comparisonCompleted.emit(results)
                return
            
            # Paso 4: Obtener imágenes correspondientes y comparar detalladamente
            self.progressUpdated.emit(60, "Comparando con candidatos encontrados...")
            
            detailed_results = []
            total_candidates = len(similar_vectors)
            
            for i, (vector_idx, distance) in enumerate(similar_vectors):
                if self.should_stop:
                    return
                
                progress = 60 + (i * 30 / total_candidates)
                self.progressUpdated.emit(int(progress), f"Comparando candidato {i+1}/{total_candidates}...")
                
                try:
                    # Obtener imagen de base de datos por índice vectorial
                    db_image = self._get_image_by_vector_index(db, vector_idx)
                    if not db_image or not os.path.exists(db_image.file_path):
                        continue
                    
                    # Comparación detallada usando UnifiedMatcher
                    comparison_result = matcher.compare_image_files(query_path, db_image.file_path)
                    
                    if comparison_result.similarity_score >= similarity_threshold:
                        # Obtener información del caso asociado
                        case_info = db.get_case_by_id(db_image.case_id) if db_image.case_id else None
                        
                        result_item = {
                            'id': f'DB-{db_image.id}',
                            'path': db_image.file_path,
                            'cmc_score': comparison_result.similarity_score,
                            'afte_conclusion': self._determine_afte_conclusion(comparison_result.similarity_score),
                            'case_number': case_info.case_number if case_info else f'CASE-{db_image.case_id}',
                            'weapon_type': case_info.weapon_type if case_info else 'Unknown',
                            'date_added': db_image.date_added,
                            'metadata': {
                                'caliber': case_info.caliber if case_info else 'Unknown',
                                'evidence_type': db_image.evidence_type,
                                'matches_found': comparison_result.matches_count,
                                'algorithm_used': config.algorithm.value,
                                'distance': distance,
                                'image_id': db_image.id
                            }
                        }
                        detailed_results.append(result_item)
                        
                except Exception as e:
                    self.logger.warning(f"Error comparando imagen {vector_idx}: {e}")
                    continue
            
            # Paso 5: Ordenar y formatear resultados finales
            self.progressUpdated.emit(90, "Ordenando resultados por relevancia...")
            
            # Ordenar por score CMC descendente
            detailed_results.sort(key=lambda x: x['cmc_score'], reverse=True)
            
            # Limitar a max_results
            detailed_results = detailed_results[:max_results]
            
            # Formatear resultado final
            final_results = self._format_search_results(
                query_path=query_path,
                evidence_type=evidence_type,
                total_searched=db.get_database_stats().get('total_images', 0),
                candidates_found=total_candidates,
                results=detailed_results
            )
            
            self.progressUpdated.emit(100, "Búsqueda completada")
            self.comparisonCompleted.emit(final_results)
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda balística real: {e}")
            self.comparisonError.emit(f"Error en búsqueda: {str(e)}")
    
    def _features_to_vector(self, features):
        """Convierte características extraídas a vector para FAISS"""
        # Implementación simplificada - en producción sería más sofisticada
        if hasattr(features, 'descriptors') and features.descriptors is not None:
            # Para ORB/SIFT descriptors
            descriptors = features.descriptors
            if len(descriptors) > 0:
                # Usar estadísticas de los descriptors como vector
                mean_desc = np.mean(descriptors, axis=0)
                std_desc = np.std(descriptors, axis=0)
                vector = np.concatenate([mean_desc, std_desc])
                return vector.astype(np.float32).reshape(1, -1)
        
        # Fallback: vector aleatorio normalizado
        return np.random.rand(1, 128).astype(np.float32)
    
    def _get_image_by_vector_index(self, db, vector_idx):
        """Obtiene imagen de BD por índice vectorial"""
        # Implementación simplificada - mapear índice a ID de imagen
        # En producción habría una tabla de mapeo
        try:
            return db.get_image_by_id(vector_idx + 1)  # Asumiendo mapeo secuencial
        except:
            return None
    
    def _determine_afte_conclusion(self, similarity_score):
        """Determina conclusión AFTE basada en score de similitud"""
        if similarity_score >= 0.9:
            return "Identification"
        elif similarity_score >= 0.7:
            return "Probable"
        elif similarity_score >= 0.5:
            return "Possible"
        else:
            return "Inconclusive"
    
    def _format_empty_search_results(self, query_path, evidence_type):
        """Formatea resultado vacío cuando no hay coincidencias"""
        return {
            'mode': 'database',
            'evidence_type': evidence_type,
            'query_image': query_path,
            'total_searched': 0,
            'candidates_found': 0,
            'high_confidence_matches': 0,
            'search_time': 0.1,
            'results': []
        }
    
    def _format_search_results(self, query_path, evidence_type, total_searched, candidates_found, results):
        """Formatea resultados finales de búsqueda"""
        high_confidence = len([r for r in results if r['cmc_score'] >= 0.8])
        
        return {
            'mode': 'database',
            'evidence_type': evidence_type,
            'query_image': query_path,
            'total_searched': total_searched,
            'candidates_found': candidates_found,
            'high_confidence_matches': high_confidence,
            'search_time': 2.5,  # Tiempo simulado
            'results': results
        }

class CMCVisualizationWidget(QWidget):
    """Widget especializado para visualizar curvas CMC y análisis estadístico"""
    
    def __init__(self):
        super().__init__()
        self.cmc_data = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Análisis CMC (Congruent Matching Cells)")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Área de visualización
        self.visualization_area = QLabel("Cargar datos CMC para visualizar curva")
        self.visualization_area.setMinimumHeight(200)
        self.visualization_area.setStyleSheet("border: 1px solid #ccc; background: #f9f9f9;")
        self.visualization_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_area)
        
    def update_cmc_data(self, cmc_data: dict):
        """Actualiza la visualización con nuevos datos CMC"""
        self.cmc_data = cmc_data
        self.render_cmc_visualization()
        
    def setup_connections(self):
        """Configura las conexiones de señales y slots para el widget CMC"""
        # Por ahora no hay conexiones específicas que configurar
        # Este método se puede expandir en el futuro si se necesitan conexiones
        pass
        
    def render_cmc_visualization(self):
        """Renderiza la visualización CMC"""
        if not self.cmc_data:
            return
            
        # Crear pixmap para dibujar
        pixmap = QPixmap(400, 200)
        pixmap.fill(QColor(255, 255, 255))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dibujar curva CMC simulada
        pen = QPen(QColor(0, 120, 215), 2)
        painter.setPen(pen)
        
        # Simular curva CMC
        points = []
        for i in range(50):
            x = i * 8
            y = 180 - (self.cmc_data.get('cmc_score', 0.8) * 160 * (1 - np.exp(-i/10)))
            points.append((x, y))
            
        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]), 
                           int(points[i+1][0]), int(points[i+1][1]))
        
        painter.end()
        self.visualization_area.setPixmap(pixmap)

class ComparisonTab(QWidget):
    """Pestaña de análisis comparativo balístico especializada"""
    
    comparisonCompleted = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_mode = 'direct'
        self.comparison_data = {}
        self.comparison_worker = None
        self.selected_db_result = None
        
        # Navigation state variables
        self.current_step = 0
        
        # Inicializar widgets faltantes para evitar AttributeError
        self.cmc_visualization = CMCVisualizationWidget()
        
        # Inicializar ballistic_features_text
        self.ballistic_features_text = QTextEdit()
        self.ballistic_features_text.setReadOnly(True)
        
        # Crear query_image_viewer si no existe
        try:
            from .image_viewer import ImageViewer
            self.query_image_viewer = ImageViewer()
        except ImportError:
            # Fallback a QLabel si ImageViewer no está disponible
            from PyQt5.QtWidgets import QLabel
            self.query_image_viewer = QLabel("Image Viewer")
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configura la interfaz especializada para análisis balístico"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Navigation buttons (will be added to individual tabs)
        self.setup_navigation_buttons()
        
        # Contenido principal con modos especializados
        self.setup_ballistic_mode_tabs()
        main_layout.addWidget(self.mode_tabs)
        
    def setup_ballistic_mode_tabs(self):
        """Configura las pestañas especializadas para análisis balístico"""
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setProperty("class", "mode-tabs")
        
        # Modo 1: Comparación Directa Balística
        self.direct_tab = self.create_direct_ballistic_tab()
        self.mode_tabs.addTab(self.direct_tab, " Comparación Directa")
        
        # Modo 2: Búsqueda en Base de Datos Balística
        self.database_tab = self.create_database_ballistic_tab()
        self.mode_tabs.addTab(self.database_tab, " Búsqueda en BD")
        
    def setup_navigation_buttons(self):
        """Configura los botones de navegación"""
        self.navigation_frame = QFrame()
        nav_layout = QHBoxLayout(self.navigation_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        # Botón anterior
        self.prev_button = QPushButton("Anterior")
        self.prev_button.setProperty("class", "nav-button")
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        # Espaciador
        nav_layout.addStretch()
        
        # Botón reiniciar
        self.reset_button = QPushButton("Reiniciar")
        self.reset_button.setProperty("class", "nav-button reset")
        nav_layout.addWidget(self.reset_button)
        
        # Espaciador
        nav_layout.addStretch()
        
        # Botón siguiente
        self.next_button = QPushButton("Siguiente")
        self.next_button.setProperty("class", "nav-button")
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
    def create_direct_ballistic_tab(self) -> QWidget:
        """Crea la pestaña de comparación directa balística con paneles adaptativos mejorados"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        
        # Crear splitter principal para paneles adaptativos
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)  # Evitar que se colapsen completamente
        main_splitter.setHandleWidth(8)  # Ancho del divisor más visible
        main_splitter.setOpaqueResize(False)  # Redimensionamiento suave
        
        # Panel izquierdo - Flujo de trabajo scrolleable con adaptabilidad mejorada
        left_panel = QWidget()
        left_panel.setMinimumWidth(280)  # Ancho mínimo optimizado
        left_panel.setMaximumWidth(800)  # Ancho máximo más amplio para pantallas grandes
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Panel de configuración scrolleable
        config_panel = self.create_direct_ballistic_config_panel()
        left_layout.addWidget(config_panel)
        
        # Botones de navegación debajo del panel izquierdo
        left_layout.addWidget(self.navigation_frame)
        
        # Panel derecho - Visualizaciones y resultados scrolleables con adaptabilidad mejorada
        right_panel = QWidget()
        right_panel.setMinimumWidth(400)  # Ancho mínimo optimizado para visualizaciones
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Crear splitter vertical para visualizaciones y resultados
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.setChildrenCollapsible(False)
        vertical_splitter.setHandleWidth(6)  # Divisor vertical más sutil
        vertical_splitter.setOpaqueResize(False)  # Redimensionamiento suave
        
        # Panel de visualización con adaptabilidad
        visual_panel = self.create_direct_ballistic_visual_panel()
        visual_panel.setMinimumHeight(300)  # Altura mínima para visualizaciones
        visual_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        vertical_splitter.addWidget(visual_panel)
        
        # Panel de resultados con adaptabilidad mejorada
        results_panel = QWidget()
        results_panel.setMinimumHeight(200)  # Altura mínima para resultados
        results_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(5, 5, 5, 5)
        
        results_title = QLabel("Resultados del Análisis")
        results_title.setProperty("class", "subtitle")
        results_layout.addWidget(results_title)
        
        # Área scrolleable para resultados
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        results_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        results_content = QWidget()
        results_content_layout = QVBoxLayout(results_content)
        results_content_layout.addWidget(QLabel("Los resultados aparecerán aquí después del análisis"))
        results_scroll.setWidget(results_content)
        results_layout.addWidget(results_scroll)
        
        vertical_splitter.addWidget(results_panel)
        
        # Configurar proporciones del splitter vertical adaptativas
        vertical_splitter.setSizes([500, 300])  # Tamaños iniciales más equilibrados
        vertical_splitter.setStretchFactor(0, 2)  # Panel de visualización
        vertical_splitter.setStretchFactor(1, 1)  # Panel de resultados
        
        right_layout.addWidget(vertical_splitter)
        
        # Agregar paneles al splitter principal
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Configurar proporciones del splitter principal adaptativas
        # Calcular tamaños basados en el ancho disponible
        initial_left_size = 350  # Tamaño inicial del panel izquierdo
        initial_right_size = 650  # Tamaño inicial del panel derecho
        main_splitter.setSizes([initial_left_size, initial_right_size])
        main_splitter.setStretchFactor(0, 1)  # Panel izquierdo - menos flexible
        main_splitter.setStretchFactor(1, 2)  # Panel derecho - más flexible
        
        # Conectar señales para adaptabilidad dinámica
        main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        vertical_splitter.splitterMoved.connect(self._on_vertical_splitter_moved)
        
        main_layout.addWidget(main_splitter)
        
        # Guardar referencias para uso posterior
        self.main_splitter = main_splitter
        self.vertical_splitter = vertical_splitter
        self.results_panel = results_panel
        self.left_panel = left_panel
        self.right_panel = right_panel
        
        # Configurar redimensionamiento automático
        self._setup_adaptive_resizing()
        
        return tab

    def _on_main_splitter_moved(self, pos: int, index: int):
        """Maneja el movimiento del splitter principal para adaptabilidad dinámica"""
        try:
            sizes = self.main_splitter.sizes()
            total_width = sum(sizes)
            
            if total_width > 0:
                left_ratio = sizes[0] / total_width
                right_ratio = sizes[1] / total_width
                
                # Ajustar políticas de tamaño basadas en las proporciones
                if left_ratio > 0.6:  # Panel izquierdo muy grande
                    self.left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
                    self.right_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
                elif right_ratio > 0.7:  # Panel derecho muy grande
                    self.left_panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
                    self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                else:  # Proporciones equilibradas
                    self.left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
                    self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception as e:
            print(f"Error en _on_main_splitter_moved: {e}")

    def _on_vertical_splitter_moved(self, pos: int, index: int):
        """Maneja el movimiento del splitter vertical para adaptabilidad dinámica"""
        try:
            if hasattr(self, 'vertical_splitter'):
                sizes = self.vertical_splitter.sizes()
                total_height = sum(sizes)
                
                if total_height > 0:
                    visual_ratio = sizes[0] / total_height
                    results_ratio = sizes[1] / total_height
                    
                    # Ajustar visibilidad y políticas basadas en las proporciones
                    if results_ratio < 0.15:  # Panel de resultados muy pequeño
                        self.results_panel.setVisible(True)  # Mantener visible pero pequeño
                    elif visual_ratio < 0.2:  # Panel de visualización muy pequeño
                        # Restaurar tamaño mínimo del panel de visualización
                        self.vertical_splitter.setSizes([300, sizes[1]])
        except Exception as e:
            print(f"Error en _on_vertical_splitter_moved: {e}")

    def _setup_adaptive_resizing(self):
        """Configura el redimensionamiento adaptativo automático"""
        try:
            # Configurar timer para redimensionamiento suave
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self._apply_adaptive_sizing)
            
            # Conectar eventos de redimensionamiento de ventana
            if hasattr(self, 'parent') and self.parent():
                parent = self.parent()
                while parent and not hasattr(parent, 'resizeEvent'):
                    parent = parent.parent()
                if parent:
                    # Guardar el método original de redimensionamiento
                    original_resize = parent.resizeEvent
                    
                    def adaptive_resize_event(event):
                        original_resize(event)
                        self.resize_timer.start(100)  # Delay para evitar múltiples llamadas
                    
                    parent.resizeEvent = adaptive_resize_event
        except Exception as e:
            print(f"Error configurando redimensionamiento adaptativo: {e}")

    def _apply_adaptive_sizing(self):
        """Aplica el dimensionamiento adaptativo basado en el tamaño de la ventana"""
        try:
            if hasattr(self, 'main_splitter') and self.main_splitter:
                # Obtener el ancho total disponible
                total_width = self.width()
                
                if total_width > 1200:  # Pantalla grande
                    # Dar más espacio al panel derecho para visualizaciones
                    left_size = int(total_width * 0.35)
                    right_size = int(total_width * 0.65)
                elif total_width > 800:  # Pantalla mediana
                    # Proporciones equilibradas
                    left_size = int(total_width * 0.4)
                    right_size = int(total_width * 0.6)
                else:  # Pantalla pequeña
                    # Priorizar panel de configuración
                    left_size = int(total_width * 0.45)
                    right_size = int(total_width * 0.55)
                
                # Aplicar los nuevos tamaños suavemente
                current_sizes = self.main_splitter.sizes()
                if abs(current_sizes[0] - left_size) > 50:  # Solo si hay diferencia significativa
                    self.main_splitter.setSizes([left_size, right_size])
        except Exception as e:
            print(f"Error aplicando dimensionamiento adaptativo: {e}")
        
    def create_direct_ballistic_config_panel(self) -> QWidget:
        """Crea el panel de configuración para comparación directa balística"""
        # Crear scroll area para hacer el panel desplazable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de pasos balísticos
        steps = ["Cargar Evidencias", "Datos del Caso", "Metadatos NIST", "Config. Análisis", "Análisis CMC", "Conclusión AFTE"]
        self.direct_step_indicator = StepIndicator(steps)
        layout.addWidget(self.direct_step_indicator)
        
        # Paso 1: Cargar evidencias balísticas
        evidence_group = QGroupBox("Paso 1: Cargar Evidencias Balísticas")
        evidence_layout = QVBoxLayout(evidence_group)
        
        # Selector de tipo de evidencia
        evidence_type_layout = QHBoxLayout()
        evidence_type_label = QLabel("Tipo de Evidencia:")
        evidence_type_label.setProperty("class", "body")
        evidence_type_layout.addWidget(evidence_type_label)
        
        self.evidence_type_combo = QComboBox()
        self.evidence_type_combo.addItems([
            "Casquillo (Cartridge Case)",
            "Bala (Bullet)",
            "Fragmento Balístico"
        ])
        self.evidence_type_combo.setMinimumWidth(200)
        evidence_type_layout.addWidget(self.evidence_type_combo)
        evidence_type_layout.addStretch()
        
        evidence_layout.addLayout(evidence_type_layout)
        
        # Drop zones especializadas
        drop_layout = QHBoxLayout()
        
        self.evidence_a_zone = ImageDropZone("Evidencia A", "Arrastrar primera evidencia\n(casquillo, bala, etc.)")
        drop_layout.addWidget(self.evidence_a_zone)
        
        vs_label = QLabel("VS")
        vs_label.setProperty("class", "title")
        vs_label.setAlignment(Qt.AlignCenter)
        vs_label.setFixedWidth(40)
        drop_layout.addWidget(vs_label)
        
        self.evidence_b_zone = ImageDropZone("Evidencia B", "Arrastrar segunda evidencia\n(mismo tipo)")
        drop_layout.addWidget(self.evidence_b_zone)
        
        evidence_layout.addLayout(drop_layout)
        layout.addWidget(evidence_group)
        
        # Paso 2: Datos del Caso Comparativo
        self.case_data_group = QGroupBox("Paso 2: Datos del Caso Comparativo")
        self.case_data_group.setProperty("class", "step-group")
        self.case_data_group.setEnabled(False)  # Inicialmente deshabilitado
        case_data_layout = QVBoxLayout(self.case_data_group)
        
        # Información básica del caso
        basic_info_group = QGroupBox("Información Básica")
        basic_layout = QGridLayout(basic_info_group)
        
        # Campos obligatorios
        self.case_number_edit = QLineEdit()
        self.case_number_edit.setPlaceholderText("Ej: COMP-2024-001")
        basic_layout.addWidget(QLabel("Número de Caso:*"), 0, 0)
        basic_layout.addWidget(self.case_number_edit, 0, 1)
        
        self.evidence_a_id_edit = QLineEdit()
        self.evidence_a_id_edit.setPlaceholderText("Ej: EV-001-A")
        basic_layout.addWidget(QLabel("ID Evidencia A:*"), 1, 0)
        basic_layout.addWidget(self.evidence_a_id_edit, 1, 1)
        
        self.evidence_b_id_edit = QLineEdit()
        self.evidence_b_id_edit.setPlaceholderText("Ej: EV-001-B")
        basic_layout.addWidget(QLabel("ID Evidencia B:*"), 2, 0)
        basic_layout.addWidget(self.evidence_b_id_edit, 2, 1)
        
        self.examiner_edit = QLineEdit()
        self.examiner_edit.setPlaceholderText("Nombre del perito balístico")
        basic_layout.addWidget(QLabel("Examinador:*"), 3, 0)
        basic_layout.addWidget(self.examiner_edit, 3, 1)
        
        case_data_layout.addWidget(basic_info_group)
        
        # Información del arma (opcional)
        weapon_info_group = QGroupBox("Información de Armas (Opcional)")
        weapon_layout = QGridLayout(weapon_info_group)
        
        # Arma A
        weapon_layout.addWidget(QLabel("Arma A:"), 0, 0, 1, 2)
        
        self.weapon_a_make_edit = QLineEdit()
        self.weapon_a_make_edit.setPlaceholderText("Marca del Arma A")
        weapon_layout.addWidget(QLabel("Marca:"), 1, 0)
        weapon_layout.addWidget(self.weapon_a_make_edit, 1, 1)
        
        self.weapon_a_model_edit = QLineEdit()
        self.weapon_a_model_edit.setPlaceholderText("Modelo del Arma A")
        weapon_layout.addWidget(QLabel("Modelo:"), 2, 0)
        weapon_layout.addWidget(self.weapon_a_model_edit, 2, 1)
        
        self.caliber_a_edit = QLineEdit()
        self.caliber_a_edit.setPlaceholderText("Calibre del Arma A")
        weapon_layout.addWidget(QLabel("Calibre:"), 3, 0)
        weapon_layout.addWidget(self.caliber_a_edit, 3, 1)
        
        # Separador entre armas
        weapon_separator = QFrame()
        weapon_separator.setFrameShape(QFrame.HLine)
        weapon_separator.setFrameShadow(QFrame.Sunken)
        weapon_layout.addWidget(weapon_separator, 4, 0, 1, 2)
        
        # Arma B
        weapon_layout.addWidget(QLabel("Arma B:"), 5, 0, 1, 2)
        
        self.weapon_b_make_edit = QLineEdit()
        self.weapon_b_make_edit.setPlaceholderText("Marca del Arma B")
        weapon_layout.addWidget(QLabel("Marca:"), 6, 0)
        weapon_layout.addWidget(self.weapon_b_make_edit, 6, 1)
        
        self.weapon_b_model_edit = QLineEdit()
        self.weapon_b_model_edit.setPlaceholderText("Modelo del Arma B")
        weapon_layout.addWidget(QLabel("Modelo:"), 7, 0)
        weapon_layout.addWidget(self.weapon_b_model_edit, 7, 1)
        
        self.caliber_b_edit = QLineEdit()
        self.caliber_b_edit.setPlaceholderText("Calibre del Arma B")
        weapon_layout.addWidget(QLabel("Calibre:"), 8, 0)
        weapon_layout.addWidget(self.caliber_b_edit, 8, 1)
        
        case_data_layout.addWidget(weapon_info_group)
        
        # Información adicional
        additional_info_group = QGroupBox("Información Adicional")
        additional_layout = QVBoxLayout(additional_info_group)
        
        self.case_description_edit = QTextEdit()
        self.case_description_edit.setPlaceholderText("Descripción del caso comparativo, circunstancias, etc.")
        self.case_description_edit.setMaximumHeight(80)
        additional_layout.addWidget(QLabel("Descripción del Caso:"))
        additional_layout.addWidget(self.case_description_edit)
        
        case_data_layout.addWidget(additional_info_group)
        layout.addWidget(self.case_data_group)

        # Paso 3: Metadatos NIST para Comparación
        self.nist_group = QGroupBox("Paso 3: Metadatos NIST Comparativos (Opcional)")
        self.nist_group.setProperty("class", "step-group")
        self.nist_group.setEnabled(False)  # Inicialmente deshabilitado
        nist_layout = QVBoxLayout(self.nist_group)
        
        # Checkbox para habilitar metadatos NIST
        self.enable_nist_checkbox = QCheckBox("Incluir metadatos en formato NIST para comparación balística")
        nist_layout.addWidget(self.enable_nist_checkbox)
        
        # Panel de metadatos NIST (colapsable)
        self.nist_panel = CollapsiblePanel("Configuración de Metadatos NIST Comparativos")
        
        nist_form = QFormLayout()
        
        # Información del laboratorio
        self.lab_name_edit = QLineEdit()
        self.lab_name_edit.setPlaceholderText("Nombre del laboratorio forense")
        nist_form.addRow("Laboratorio:", self.lab_name_edit)
        
        self.lab_accreditation_edit = QLineEdit()
        self.lab_accreditation_edit.setPlaceholderText("Número de acreditación")
        nist_form.addRow("Acreditación:", self.lab_accreditation_edit)
        
        # Información del equipo de captura
        self.capture_device_edit = QLineEdit()
        self.capture_device_edit.setPlaceholderText("Ej: Microscopio de comparación Leica FSC")
        nist_form.addRow("Dispositivo de Captura:", self.capture_device_edit)
        
        self.magnification_edit = QLineEdit()
        self.magnification_edit.setPlaceholderText("Ej: 40x, 100x")
        nist_form.addRow("Magnificación:", self.magnification_edit)
        
        # Condiciones de iluminación
        self.lighting_type_combo = QComboBox()
        self.lighting_type_combo.addItems([
            "Seleccionar...",
            "Luz Blanca Coaxial",
            "Luz Oblicua",
            "Luz Polarizada",
            "Luz LED Ring"
        ])
        nist_form.addRow("Tipo de Iluminación:", self.lighting_type_combo)
        
        # Información de calibración
        self.calibration_date_edit = QLineEdit()
        self.calibration_date_edit.setPlaceholderText("YYYY-MM-DD")
        nist_form.addRow("Fecha de Calibración:", self.calibration_date_edit)
        
        self.scale_factor_edit = QLineEdit()
        self.scale_factor_edit.setPlaceholderText("Ej: 0.5 μm/pixel")
        nist_form.addRow("Factor de Escala:", self.scale_factor_edit)
        
        # Crear un widget contenedor para el layout
        nist_widget = QWidget()
        nist_widget.setLayout(nist_form)
        self.nist_panel.add_content_widget(nist_widget)
        nist_layout.addWidget(self.nist_panel)
        
        layout.addWidget(self.nist_group)

        # Paso 4: Configuración de Análisis Comparativo
        self.processing_group = QGroupBox("Paso 4: Configuración de Análisis Comparativo")
        self.processing_group.setProperty("class", "step-group")
        self.processing_group.setEnabled(False)  # Inicialmente deshabilitado
        processing_layout = QVBoxLayout(self.processing_group)
        
        # Configuración básica (siempre visible)
        basic_frame = QFrame()
        basic_frame.setProperty("class", "config-basic")
        basic_processing_layout = QFormLayout(basic_frame)
        
        # Nivel de análisis balístico
        self.analysis_level_combo = QComboBox()
        self.analysis_level_combo.addItems([
            "Básico - Comparación de características principales",
            "Intermedio - Análisis detallado + métricas NIST",
            "Avanzado - Análisis completo + comparación automática",
            "Forense - Análisis exhaustivo + conclusiones AFTE"
        ])
        basic_processing_layout.addRow("Nivel de Análisis:", self.analysis_level_combo)
        
        # Prioridad del procesamiento
        self.priority_combo = QComboBox()
        self.priority_combo.addItems([
            "Normal - Procesamiento estándar",
            "Alta - Procesamiento prioritario",
            "Crítica - Procesamiento inmediato"
        ])
        basic_processing_layout.addRow("Prioridad:", self.priority_combo)
        
        processing_layout.addWidget(basic_frame)
        
        # Opciones avanzadas (colapsables)
        self.advanced_panel = CollapsiblePanel("Opciones Avanzadas de Análisis Comparativo")
        
        advanced_content = QWidget()
        advanced_layout = QVBoxLayout(advanced_content)
        
        # Características balísticas a comparar
        ballistic_features_group = QGroupBox("Características Balísticas a Comparar")
        ballistic_features_layout = QVBoxLayout(ballistic_features_group)
        
        self.compare_firing_pin_cb = QCheckBox("Comparación de marcas de percutor (Firing Pin)")
        self.compare_breech_face_cb = QCheckBox("Análisis comparativo de cara de recámara (Breech Face)")
        self.compare_extractor_cb = QCheckBox("Comparación de marcas de extractor y eyector")
        self.compare_striations_cb = QCheckBox("Comparación de patrones de estriado (para balas)")
        self.compare_land_groove_cb = QCheckBox("Análisis comparativo de campos y estrías")
        
        ballistic_features_layout.addWidget(self.compare_firing_pin_cb)
        ballistic_features_layout.addWidget(self.compare_breech_face_cb)
        ballistic_features_layout.addWidget(self.compare_extractor_cb)
        ballistic_features_layout.addWidget(self.compare_striations_cb)
        ballistic_features_layout.addWidget(self.compare_land_groove_cb)
        
        advanced_layout.addWidget(ballistic_features_group)
        
        # Validación NIST para comparación
        nist_validation_group = QGroupBox("Validación NIST Comparativa")
        nist_validation_layout = QVBoxLayout(nist_validation_group)
        
        self.nist_quality_check_cb = QCheckBox("Verificación de calidad de ambas imágenes")
        self.nist_authenticity_cb = QCheckBox("Validación de autenticidad comparativa")
        self.nist_compression_cb = QCheckBox("Análisis de compresión de ambas muestras")
        self.nist_metadata_cb = QCheckBox("Validación de metadatos comparativos")
        
        nist_validation_layout.addWidget(self.nist_quality_check_cb)
        nist_validation_layout.addWidget(self.nist_authenticity_cb)
        nist_validation_layout.addWidget(self.nist_compression_cb)
        nist_validation_layout.addWidget(self.nist_metadata_cb)
        
        advanced_layout.addWidget(nist_validation_group)
        
        # Conclusiones AFTE para comparación
        afte_group = QGroupBox("Conclusiones AFTE Comparativas")
        afte_layout = QVBoxLayout(afte_group)
        
        self.generate_afte_cb = QCheckBox("Generar conclusiones AFTE automáticas para comparación")
        self.afte_confidence_cb = QCheckBox("Calcular nivel de confianza comparativo")
        self.afte_comparison_cb = QCheckBox("Comparación con base de datos de casos similares")
        
        afte_layout.addWidget(self.generate_afte_cb)
        afte_layout.addWidget(self.afte_confidence_cb)
        afte_layout.addWidget(self.afte_comparison_cb)
        
        advanced_layout.addWidget(afte_group)
        
        # Procesamiento de imagen balística comparativo
        image_processing_group = QGroupBox("Procesamiento de Imagen Comparativo")
        image_processing_layout = QVBoxLayout(image_processing_group)
        
        self.noise_reduction_cb = QCheckBox("Reducción de ruido especializada en ambas muestras")
        self.contrast_enhancement_cb = QCheckBox("Mejora de contraste para marcas comparativas")
        self.edge_detection_cb = QCheckBox("Detección de bordes de características comparativas")
        self.morphological_cb = QCheckBox("Operaciones morfológicas comparativas")
        
        image_processing_layout.addWidget(self.noise_reduction_cb)
        image_processing_layout.addWidget(self.contrast_enhancement_cb)
        image_processing_layout.addWidget(self.edge_detection_cb)
        image_processing_layout.addWidget(self.morphological_cb)
        
        advanced_layout.addWidget(image_processing_group)
        
        # Crear un widget contenedor para el layout avanzado
        self.advanced_panel.add_content_widget(advanced_content)
        processing_layout.addWidget(self.advanced_panel)
        
        layout.addWidget(self.processing_group)
        
        # Paso 5: Configuración de análisis balístico (original)
        self.ballistic_config_group = QGroupBox("Paso 5: Configuración CMC y AFTE")
        self.ballistic_config_group.setEnabled(False)  # Inicialmente deshabilitado
        ballistic_config_layout = QFormLayout(self.ballistic_config_group)
        
        # Método de análisis balístico
        self.ballistic_method_combo = QComboBox()
        self.ballistic_method_combo.addItems([
            "CMC (Congruent Matching Cells)",
            "Análisis de Características Individuales",
            "Correlación de Patrones de Estriado",
            "Análisis Multiespectal Combinado"
        ])
        ballistic_config_layout.addRow("Método de Análisis:", self.ballistic_method_combo)
        
        # Umbral de correlación CMC
        self.cmc_threshold_slider = QSlider(Qt.Horizontal)
        self.cmc_threshold_slider.setRange(50, 95)
        self.cmc_threshold_slider.setValue(75)
        self.cmc_threshold_label = QLabel("0.75")
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.cmc_threshold_slider)
        threshold_layout.addWidget(self.cmc_threshold_label)
        ballistic_config_layout.addRow("Umbral CMC:", threshold_layout)
        
        # Criterios AFTE
        afte_group = QGroupBox("Criterios AFTE")
        afte_layout = QVBoxLayout(afte_group)
        
        self.afte_identification_rb = QRadioButton("Identification (≥85% CMC)")
        self.afte_inconclusive_rb = QRadioButton("Inconclusive (70-84% CMC)")
        self.afte_elimination_rb = QRadioButton("Elimination (<70% CMC)")
        self.afte_auto_rb = QRadioButton("Determinación Automática")
        self.afte_auto_rb.setChecked(True)
        
        afte_layout.addWidget(self.afte_identification_rb)
        afte_layout.addWidget(self.afte_inconclusive_rb)
        afte_layout.addWidget(self.afte_elimination_rb)
        afte_layout.addWidget(self.afte_auto_rb)
        
        ballistic_config_layout.addRow(afte_group)
        layout.addWidget(self.ballistic_config_group)
        
        # Opciones avanzadas balísticas
        self.advanced_ballistic_panel = CollapsiblePanel("Opciones Avanzadas de Análisis Balístico")
        
        advanced_content = QWidget()
        advanced_layout = QVBoxLayout(advanced_content)
        
        # Características balísticas específicas
        features_group = QGroupBox("Características a Analizar")
        features_layout = QVBoxLayout(features_group)
        
        self.analyze_firing_pin_cb = QCheckBox("Marcas de percutor (Firing Pin)")
        self.analyze_breech_face_cb = QCheckBox("Patrones de cara de recámara (Breech Face)")
        self.analyze_extractor_cb = QCheckBox("Marcas de extractor/eyector")
        self.analyze_striations_cb = QCheckBox("Patrones de estriado (para balas)")
        self.analyze_land_groove_cb = QCheckBox("Análisis de campos y estrías")
        
        self.analyze_firing_pin_cb.setChecked(True)
        self.analyze_breech_face_cb.setChecked(True)
        self.analyze_extractor_cb.setChecked(True)
        
        features_layout.addWidget(self.analyze_firing_pin_cb)
        features_layout.addWidget(self.analyze_breech_face_cb)
        features_layout.addWidget(self.analyze_extractor_cb)
        features_layout.addWidget(self.analyze_striations_cb)
        features_layout.addWidget(self.analyze_land_groove_cb)
        
        advanced_layout.addWidget(features_group)
        
        # Cambiar nist_group a self.nist_group para control de navegación
        self.nist_group = QGroupBox("Validación NIST")
        # Inicialmente deshabilitado - controlado por navegación de pasos
        self.nist_group.setEnabled(False)
        nist_layout = QVBoxLayout(self.nist_group)
        
        self.nist_quality_validation_cb = QCheckBox("Validación de calidad de imagen NIST")
        self.nist_metadata_validation_cb = QCheckBox("Validación de metadatos NIST")
        self.nist_chain_custody_cb = QCheckBox("Verificación de cadena de custodia")
        
        self.nist_quality_validation_cb.setChecked(True)
        
        nist_layout.addWidget(self.nist_quality_validation_cb)
        nist_layout.addWidget(self.nist_metadata_validation_cb)
        nist_layout.addWidget(self.nist_chain_custody_cb)
        
        advanced_layout.addWidget(self.nist_group)
        
        # Agregar el grupo de configuración balística (Paso 5)
        advanced_layout.addWidget(self.ballistic_config_group)
        
        # Deep Learning (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            dl_group = QGroupBox("Análisis con Deep Learning")
            dl_group.setProperty("class", "dl-group")
            dl_layout = QVBoxLayout(dl_group)
            
            self.enable_dl_comparison_cb = QCheckBox("Habilitar análisis con Deep Learning")
            self.enable_dl_comparison_cb.setProperty("class", "dl-checkbox")
            dl_layout.addWidget(self.enable_dl_comparison_cb)
            
            # Selector de modelo
            model_layout = QFormLayout()
            self.dl_comparison_model_combo = QComboBox()
            self.dl_comparison_model_combo.setProperty("class", "dl-combo")
            self.dl_comparison_model_combo.addItems([
                "SiameseNetwork - Comparación de similitud",
                "BallisticCNN - Extracción de características",
                "Ensemble - Combinación de modelos"
            ])
            self.dl_comparison_model_combo.setEnabled(False)
            model_layout.addRow("Modelo DL:", self.dl_comparison_model_combo)
            
            # Umbral de confianza
            self.dl_confidence_spin = QDoubleSpinBox()
            self.dl_confidence_spin.setProperty("class", "dl-spin")
            self.dl_confidence_spin.setRange(0.1, 1.0)
            self.dl_confidence_spin.setSingleStep(0.05)
            self.dl_confidence_spin.setValue(0.85)
            self.dl_confidence_spin.setEnabled(False)
            model_layout.addRow("Confianza mínima:", self.dl_confidence_spin)
            
            dl_layout.addLayout(model_layout)
            
            # Botón de configuración avanzada
            self.dl_advanced_comparison_button = QPushButton("⚙️ Configuración Avanzada")
            self.dl_advanced_comparison_button.setProperty("class", "dl-advanced")
            self.dl_advanced_comparison_button.setEnabled(False)
            self.dl_advanced_comparison_button.clicked.connect(self.open_comparison_model_selector)
            dl_layout.addWidget(self.dl_advanced_comparison_button)
            
            # Conectar señales
            self.enable_dl_comparison_cb.toggled.connect(self.toggle_dl_comparison_options)
            advanced_layout.addWidget(dl_group)
            
            # Conectar señales
            self.enable_dl_comparison_cb.toggled.connect(self.dl_comparison_model_combo.setEnabled)
            self.enable_dl_comparison_cb.toggled.connect(self.dl_confidence_spin.setEnabled)
        
        self.advanced_ballistic_panel.add_content_widget(advanced_content)
        layout.addWidget(self.advanced_ballistic_panel)
        
        # Botón de análisis
        self.analyze_button = QPushButton(" Iniciar Análisis Balístico")
        self.analyze_button.setProperty("class", "primary-button")
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # Progress card
        self.direct_progress_card = ProgressCard("Análisis en progreso...")
        self.direct_progress_card.hide()
        layout.addWidget(self.direct_progress_card)
        
        # Botones de configuración
        config_buttons_layout = QHBoxLayout()
        
        self.save_config_button = QPushButton(" Guardar Configuración")
        self.save_config_button.setProperty("class", "secondary-button")
        config_buttons_layout.addWidget(self.save_config_button)
        
        self.reset_config_button = QPushButton(" Reiniciar Configuración")
        self.reset_config_button.setProperty("class", "secondary-button")
        config_buttons_layout.addWidget(self.reset_config_button)
        
        layout.addLayout(config_buttons_layout)
        
        layout.addStretch()
        
        # Configurar el scroll area con el panel
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def create_direct_ballistic_visual_panel(self) -> QWidget:
        """Crea el panel de visualización para comparación directa balística"""
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título
        title = QLabel("Resultados de Comparación Balística")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Área de resultados con tabs
        self.results_tabs = QTabWidget()
        
        # Tab 1: Visor Sincronizado
        sync_tab = QWidget()
        sync_layout = QVBoxLayout(sync_tab)
        
        self.synchronized_viewer = SynchronizedViewer()
        sync_layout.addWidget(self.synchronized_viewer)
        
        self.results_tabs.addTab(sync_tab, " Visor Sincronizado")
        
        # Tab 2: Alineación Asistida
        alignment_tab = QWidget()
        alignment_layout = QVBoxLayout(alignment_tab)
        
        self.assisted_alignment = AssistedAlignmentWidget()
        alignment_layout.addWidget(self.assisted_alignment)
        
        self.results_tabs.addTab(alignment_tab, " Alineación Asistida")
        
        # Tab 3: Visualización CMC Interactiva
        cmc_tab = QWidget()
        cmc_layout = QVBoxLayout(cmc_tab)
        
        self.interactive_cmc = InteractiveCMCWidget()
        cmc_layout.addWidget(self.interactive_cmc)
        
        self.results_tabs.addTab(cmc_tab, " Análisis CMC")
        
        # Tab 4: Coincidencias Interactivas
        matching_tab = QWidget()
        matching_layout = QVBoxLayout(matching_tab)
        
        self.interactive_matching = InteractiveMatchingWidget()
        matching_layout.addWidget(self.interactive_matching)
        
        self.results_tabs.addTab(matching_tab, " Coincidencias")
        
        # Tab 5: Mapa de Correlación
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        
        self.correlation_heatmap = CorrelationHeatmapWidget()
        heatmap_layout.addWidget(self.correlation_heatmap)
        
        self.results_tabs.addTab(heatmap_tab, " Mapa de Correlación")
        
        # Tab 6: Conclusión AFTE
        conclusion_tab = QWidget()
        conclusion_layout = QVBoxLayout(conclusion_tab)
        
        self.dynamic_results = DynamicResultsPanel()
        conclusion_layout.addWidget(self.dynamic_results)
        
        self.results_tabs.addTab(conclusion_tab, " Conclusión AFTE")
        
        self.results_tabs.addTab(conclusion_tab, " Conclusión")
        
        layout.addWidget(self.results_tabs)
        
        return panel
        
    def create_database_ballistic_tab(self) -> QWidget:
        """Crea la pestaña de búsqueda en base de datos balística con paneles adaptativos mejorados"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        
        # Crear splitter principal para paneles adaptativos
        database_splitter = QSplitter(Qt.Horizontal)
        database_splitter.setChildrenCollapsible(False)  # Evitar que se colapsen completamente
        database_splitter.setHandleWidth(8)  # Ancho del divisor más visible
        
        # Panel izquierdo - Configuración de búsqueda scrolleable
        search_panel = self.create_database_ballistic_config_panel()
        search_panel.setMinimumWidth(300)  # Ancho mínimo más flexible
        search_panel.setMaximumWidth(600)  # Ancho máximo más amplio
        
        # Panel derecho - Resultados de búsqueda scrolleables
        results_panel = self.create_database_ballistic_results_panel()
        results_panel.setMinimumWidth(350)  # Ancho mínimo más flexible
        
        # Agregar paneles al splitter
        database_splitter.addWidget(search_panel)
        database_splitter.addWidget(results_panel)
        
        # Configurar proporciones del splitter (40% izquierda, 60% derecha)
        database_splitter.setSizes([400, 600])
        database_splitter.setStretchFactor(0, 2)  # Panel izquierdo más flexible para redimensionar
        database_splitter.setStretchFactor(1, 3)  # Panel derecho menos flexible pero más espacio
        
        main_layout.addWidget(database_splitter)
        
        # Guardar referencia para uso posterior
        self.database_splitter = database_splitter
        
        return tab
        
    def create_database_ballistic_config_panel(self) -> QWidget:
        """Crea el panel de configuración para búsqueda en base de datos balística"""
        # Crear scroll area para hacer el panel desplazable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de pasos
        steps = ["Cargar Consulta", "Config. Búsqueda", "Buscar", "Analizar Resultados"]
        self.db_step_indicator = StepIndicator(steps)
        layout.addWidget(self.db_step_indicator)
        
        # Paso 1: Imagen de consulta
        query_group = QGroupBox("Paso 1: Evidencia de Consulta")
        query_layout = QVBoxLayout(query_group)
        
        self.query_evidence_zone = ImageDropZone("Evidencia de Consulta", "Arrastrar evidencia balística\npara buscar coincidencias")
        query_layout.addWidget(self.query_evidence_zone)
        
        layout.addWidget(query_group)
        
        # Paso 2: Configuración de búsqueda balística
        search_config_group = QGroupBox("Paso 2: Configuración de Búsqueda Balística")
        search_config_group.setEnabled(False)
        search_config_layout = QFormLayout(search_config_group)
        
        # Filtros balísticos
        self.caliber_filter_combo = QComboBox()
        self.caliber_filter_combo.addItems([
            "Todos los calibres",
            "9mm Luger",
            ".40 S&W",
            ".45 ACP",
            ".38 Special",
            ".357 Magnum",
            "5.56mm NATO",
            "7.62mm NATO"
        ])
        search_config_layout.addRow("Filtro por Calibre:", self.caliber_filter_combo)
        
        self.weapon_type_filter_combo = QComboBox()
        self.weapon_type_filter_combo.addItems([
            "Todos los tipos",
            "Pistola",
            "Revólver",
            "Rifle",
            "Escopeta",
            "Subfusil"
        ])
        search_config_layout.addRow("Tipo de Arma:", self.weapon_type_filter_combo)
        
        # Umbral de similitud
        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setRange(50, 95)
        self.similarity_threshold_slider.setValue(70)
        self.similarity_threshold_label = QLabel("0.70")
        similarity_layout = QHBoxLayout()
        similarity_layout.addWidget(self.similarity_threshold_slider)
        similarity_layout.addWidget(self.similarity_threshold_label)
        search_config_layout.addRow("Umbral de Similitud:", similarity_layout)
        
        # Número máximo de resultados
        self.max_results_spinbox = QSpinBox()
        self.max_results_spinbox.setRange(5, 100)
        self.max_results_spinbox.setValue(20)
        search_config_layout.addRow("Máx. Resultados:", self.max_results_spinbox)
        
        layout.addWidget(search_config_group)
        
        # Opciones avanzadas de búsqueda
        self.advanced_search_panel = CollapsiblePanel("Opciones Avanzadas de Búsqueda")
        
        advanced_search_content = QWidget()
        advanced_search_layout = QVBoxLayout(advanced_search_content)
        
        # Filtros temporales
        temporal_group = QGroupBox("Filtros Temporales")
        temporal_layout = QFormLayout(temporal_group)
        
        self.date_from_edit = QLineEdit()
        self.date_from_edit.setPlaceholderText("YYYY-MM-DD")
        temporal_layout.addRow("Fecha Desde:", self.date_from_edit)
        
        self.date_to_edit = QLineEdit()
        self.date_to_edit.setPlaceholderText("YYYY-MM-DD")
        temporal_layout.addRow("Fecha Hasta:", self.date_to_edit)
        
        advanced_search_layout.addWidget(temporal_group)
        
        # Filtros de metadatos
        metadata_group = QGroupBox("Filtros de Metadatos")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.case_number_filter_edit = QLineEdit()
        self.case_number_filter_edit.setPlaceholderText("Número de caso...")
        metadata_layout.addWidget(QLabel("Número de Caso:"))
        metadata_layout.addWidget(self.case_number_filter_edit)
        
        self.location_filter_edit = QLineEdit()
        self.location_filter_edit.setPlaceholderText("Ubicación...")
        metadata_layout.addWidget(QLabel("Ubicación:"))
        metadata_layout.addWidget(self.location_filter_edit)
        
        advanced_search_layout.addWidget(metadata_group)
        
        # Deep Learning para búsqueda (si está disponible)
        if DEEP_LEARNING_AVAILABLE:
            dl_search_group = QGroupBox("Búsqueda con Deep Learning")
            dl_search_group.setProperty("class", "dl-group")
            dl_search_layout = QVBoxLayout(dl_search_group)
            
            self.enable_dl_search_cb = QCheckBox("Habilitar búsqueda con Deep Learning")
            self.enable_dl_search_cb.setProperty("class", "dl-checkbox")
            dl_search_layout.addWidget(self.enable_dl_search_cb)
            
            # Configuración de modelo para búsqueda
            search_model_layout = QFormLayout()
            self.dl_search_model_combo = QComboBox()
            self.dl_search_model_combo.setProperty("class", "dl-combo")
            self.dl_search_model_combo.addItems([
                "SiameseNetwork - Búsqueda por similitud",
                "BallisticCNN - Extracción de características",
                "Ensemble - Búsqueda híbrida"
            ])
            self.dl_search_model_combo.setEnabled(False)
            search_model_layout.addRow("Modelo DL:", self.dl_search_model_combo)
            
            # Configuración específica para búsqueda
            self.dl_search_confidence_spin = QDoubleSpinBox()
            self.dl_search_confidence_spin.setProperty("class", "dl-spin")
            self.dl_search_confidence_spin.setRange(0.1, 1.0)
            self.dl_search_confidence_spin.setSingleStep(0.05)
            self.dl_search_confidence_spin.setValue(0.75)
            self.dl_search_confidence_spin.setEnabled(False)
            search_model_layout.addRow("Confianza mínima:", self.dl_search_confidence_spin)
            
            self.dl_rerank_results_cb = QCheckBox("Re-ranking con DL")
            self.dl_rerank_results_cb.setProperty("class", "dl-checkbox")
            self.dl_rerank_results_cb.setEnabled(False)
            search_model_layout.addRow("", self.dl_rerank_results_cb)
            
            dl_search_layout.addLayout(search_model_layout)
            
            # Botón de configuración avanzada para búsqueda
            self.dl_advanced_search_button = QPushButton(" Configuración Avanzada")
            self.dl_advanced_search_button.setProperty("class", "dl-advanced")
            self.dl_advanced_search_button.setEnabled(False)
            self.dl_advanced_search_button.clicked.connect(self.open_search_model_selector)
            dl_search_layout.addWidget(self.dl_advanced_search_button)
            
            advanced_search_layout.addWidget(dl_search_group)
            
            # Conectar señales
            self.enable_dl_search_cb.toggled.connect(self.toggle_dl_search_options)
        
        self.advanced_search_panel.add_content_widget(advanced_search_content)
        layout.addWidget(self.advanced_search_panel)
        
        # Botón de búsqueda
        self.search_button = QPushButton(" Buscar en Base de Datos")
        self.search_button.setProperty("class", "primary-button")
        self.search_button.setEnabled(False)
        layout.addWidget(self.search_button)
        
        # Progress card para búsqueda
        self.search_progress_card = ProgressCard("Búsqueda en progreso...")
        self.search_progress_card.hide()
        layout.addWidget(self.search_progress_card)
        
        # Botones de configuración para búsqueda
        search_config_buttons_layout = QHBoxLayout()
        
        self.save_search_config_button = QPushButton(" Guardar Configuración")
        self.save_search_config_button.setProperty("class", "secondary-button")
        search_config_buttons_layout.addWidget(self.save_search_config_button)
        
        self.reset_search_config_button = QPushButton(" Reiniciar Configuración")
        self.reset_search_config_button.setProperty("class", "secondary-button")
        search_config_buttons_layout.addWidget(self.reset_search_config_button)
        
        layout.addLayout(search_config_buttons_layout)
        
        layout.addStretch()
        
        # Configurar el scroll area con el panel
        scroll_area.setWidget(panel)
        
        return scroll_area
        
    def create_database_ballistic_results_panel(self) -> QWidget:
        """Crea el panel de resultados para búsqueda en base de datos balística con scroll"""
        # Crear scroll area para hacer el panel desplazable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        panel = QFrame()
        panel.setProperty("class", "panel")
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Título
        title = QLabel("Resultados de Búsqueda Balística")
        title.setProperty("class", "subtitle")
        layout.addWidget(title)
        
        # Widget de galería de búsqueda
        self.gallery_search = GallerySearchWidget()
        layout.addWidget(self.gallery_search)
        
        # Configurar el scroll area con el panel
        scroll_area.setWidget(panel)
        
        # Guardar referencia para uso posterior
        self.database_results_panel = scroll_area
        
        return scroll_area
        
    def setup_connections(self):
        """Configura las conexiones de señales"""
        # Conexiones de modo (solo tabs, ya no hay combo)
        self.mode_tabs.currentChanged.connect(self.on_mode_changed)
        
        # Conexiones de tipo de evidencia
        self.evidence_type_combo.currentTextChanged.connect(self.on_evidence_type_changed)
        
        # Conexiones de carga de imágenes - Modo directo
        self.evidence_a_zone.imageLoaded.connect(self.on_evidence_a_loaded)
        self.evidence_b_zone.imageLoaded.connect(self.on_evidence_b_loaded)
        
        # Conexiones de carga de imágenes - Modo búsqueda
        self.query_evidence_zone.imageLoaded.connect(self.on_query_evidence_loaded)
        
        # Conexiones de sliders
        self.cmc_threshold_slider.valueChanged.connect(self.update_cmc_threshold_label)
        self.similarity_threshold_slider.valueChanged.connect(self.update_similarity_threshold_label)
        
        # Conexiones de botones
        self.analyze_button.clicked.connect(self.start_ballistic_comparison)
        self.search_button.clicked.connect(self.start_ballistic_search)
        
        # Conexiones de botones de configuración
        self.save_config_button.clicked.connect(self.save_comparison_configuration)
        self.reset_config_button.clicked.connect(self.reset_comparison_configuration)
        self.save_search_config_button.clicked.connect(self.save_search_configuration)
        self.reset_search_config_button.clicked.connect(self.reset_search_configuration)
        
        # Conexiones de botones de navegación
        self.next_button.clicked.connect(self.next_step)
        self.prev_button.clicked.connect(self.prev_step)
        self.reset_button.clicked.connect(self.reset_workflow)
        
    def toggle_dl_comparison_options(self, enabled: bool):
        """Habilita/deshabilita las opciones de Deep Learning para comparación con manejo de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                if enabled:
                    QMessageBox.warning(
                        self, 
                        "Deep Learning No Disponible",
                        "Los módulos de Deep Learning no están disponibles.\n"
                        "Instale las dependencias necesarias:\n"
                        "pip install torch torchvision tensorflow"
                    )
                    # Desmarcar el checkbox si existe
                    if hasattr(self, 'enable_dl_comparison_cb'):
                        self.enable_dl_comparison_cb.setChecked(False)
                return
                
            # Habilitar/deshabilitar controles con verificaciones
            if hasattr(self, 'dl_comparison_model_combo'):
                self.dl_comparison_model_combo.setEnabled(enabled)
            if hasattr(self, 'dl_confidence_spin'):
                self.dl_confidence_spin.setEnabled(enabled)
            if hasattr(self, 'dl_advanced_comparison_button'):
                self.dl_advanced_comparison_button.setEnabled(enabled)
                
        except Exception as e:
            print(f"Error en toggle_dl_comparison_options: {e}")
            QMessageBox.critical(
                self, 
                "Error de Configuración",
                f"Error al configurar opciones de Deep Learning:\n{str(e)}"
            )
            # Asegurar que el checkbox esté desmarcado en caso de error
            if hasattr(self, 'enable_dl_comparison_cb'):
                self.enable_dl_comparison_cb.setChecked(False)
            
    def toggle_dl_search_options(self, enabled: bool):
        """Habilita/deshabilita las opciones de Deep Learning para búsqueda con manejo de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                if enabled:
                    QMessageBox.warning(
                        self, 
                        "Deep Learning No Disponible",
                        "Los módulos de Deep Learning no están disponibles.\n"
                        "Instale las dependencias necesarias:\n"
                        "pip install torch torchvision tensorflow"
                    )
                    # Desmarcar el checkbox si existe
                    if hasattr(self, 'enable_dl_search_cb'):
                        self.enable_dl_search_cb.setChecked(False)
                return
                
            # Habilitar/deshabilitar controles con verificaciones
            if hasattr(self, 'dl_search_model_combo'):
                self.dl_search_model_combo.setEnabled(enabled)
            if hasattr(self, 'dl_search_confidence_spin'):
                self.dl_search_confidence_spin.setEnabled(enabled)
            if hasattr(self, 'dl_rerank_results_cb'):
                self.dl_rerank_results_cb.setEnabled(enabled)
            if hasattr(self, 'dl_advanced_search_button'):
                self.dl_advanced_search_button.setEnabled(enabled)
                
        except Exception as e:
            print(f"Error en toggle_dl_search_options: {e}")
            QMessageBox.critical(
                self, 
                "Error de Configuración",
                f"Error al configurar opciones de Deep Learning:\n{str(e)}"
            )
            # Asegurar que el checkbox esté desmarcado en caso de error
            if hasattr(self, 'enable_dl_search_cb'):
                self.enable_dl_search_cb.setChecked(False)
            
    def open_comparison_model_selector(self):
        """Abre el diálogo de selección de modelos para comparación directa con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                QMessageBox.warning(
                    self, 
                    "Deep Learning No Disponible",
                    "Los módulos de Deep Learning no están disponibles.\n"
                    "Instale las dependencias necesarias:\n"
                    "pip install torch torchvision tensorflow\n\n"
                    "Verifique la instalación de las dependencias."
                )
                return
                
            # Verificar que el checkbox esté habilitado
            if hasattr(self, 'enable_dl_comparison_cb') and not self.enable_dl_comparison_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Deep Learning Deshabilitado",
                    "Debe habilitar Deep Learning para comparación antes de configurar modelos."
                )
                return
                
            # Obtener configuración actual con manejo de errores
            current_config = self.get_current_comparison_dl_config()
            
            # Importar y crear diálogo con manejo de errores
            try:
                from .model_selector_dialog import ModelSelectorDialog
                dialog = ModelSelectorDialog(self, current_config)
                dialog.modelConfigured.connect(self.on_comparison_model_configured)
                dialog.exec_()
            except ImportError as e:
                QMessageBox.critical(
                    self,
                    "Error de Importación",
                    f"No se pudo cargar el diálogo de selección de modelos:\n{str(e)}\n\n"
                    "Verifique la instalación de los módulos de Deep Learning."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error del Diálogo",
                    f"Error al mostrar el diálogo de configuración:\n{str(e)}"
                )
                
        except Exception as e:
            print(f"Error crítico en open_comparison_model_selector: {e}")
            QMessageBox.critical(
                self,
                "Error Crítico",
                f"Error inesperado al abrir configuración de modelos:\n{str(e)}"
            )
        
    def open_search_model_selector(self):
        """Abre el diálogo de selección de modelos para búsqueda en BD con manejo robusto de errores"""
        try:
            if not DEEP_LEARNING_AVAILABLE:
                QMessageBox.warning(
                    self, 
                    "Deep Learning No Disponible",
                    "Los módulos de Deep Learning no están disponibles.\n"
                    "Instale las dependencias necesarias:\n"
                    "pip install torch torchvision tensorflow\n\n"
                    "Verifique la instalación de las dependencias."
                )
                return
                
            # Verificar que el checkbox esté habilitado
            if hasattr(self, 'enable_dl_search_cb') and not self.enable_dl_search_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Deep Learning Deshabilitado",
                    "Debe habilitar Deep Learning para búsqueda antes de configurar modelos."
                )
                return
                
            # Obtener configuración actual con manejo de errores
            current_config = self.get_current_search_dl_config()
            
            # Importar y crear diálogo con manejo de errores
            try:
                from .model_selector_dialog import ModelSelectorDialog
                dialog = ModelSelectorDialog(self, current_config)
                dialog.modelConfigured.connect(self.on_search_model_configured)
                dialog.exec_()
            except ImportError as e:
                QMessageBox.critical(
                    self,
                    "Error de Importación",
                    f"No se pudo cargar el diálogo de selección de modelos:\n{str(e)}\n\n"
                    "Verifique la instalación de los módulos de Deep Learning."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error del Diálogo",
                    f"Error al mostrar el diálogo de configuración:\n{str(e)}"
                )
                
        except Exception as e:
            print(f"Error crítico en open_search_model_selector: {e}")
            QMessageBox.critical(
                self,
                "Error Crítico",
                f"Error inesperado al abrir configuración de modelos:\n{str(e)}"
            )
        
    def get_current_comparison_dl_config(self) -> dict:
        """Obtiene la configuración actual de DL para comparación con manejo robusto de errores"""
        try:
            # Verificar disponibilidad de Deep Learning
            if not DEEP_LEARNING_AVAILABLE:
                return {}
                
            # Verificar que el checkbox existe y está habilitado
            if not hasattr(self, 'enable_dl_comparison_cb') or not self.enable_dl_comparison_cb.isChecked():
                return {}
            
            config = {'enabled': True, 'task_type': 'comparison'}
            
            # Obtener configuración con verificaciones
            try:
                if hasattr(self, 'dl_comparison_model_combo'):
                    model_text = self.dl_comparison_model_combo.currentText()
                    config['model_type'] = model_text.split(' - ')[0] if ' - ' in model_text else model_text
                else:
                    config['model_type'] = 'CNN'  # Valor por defecto
                    
                if hasattr(self, 'dl_confidence_spin'):
                    config['confidence_threshold'] = self.dl_confidence_spin.value()
                else:
                    config['confidence_threshold'] = 0.85  # Valor por defecto
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración de comparación DL: {e}")
                # Usar valores por defecto en caso de error
                config.update({
                    'model_type': 'CNN',
                    'confidence_threshold': 0.85
                })
                
            return config
            
        except Exception as e:
            print(f"Error crítico obteniendo configuración de comparación DL: {e}")
            return {}  # Retornar configuración vacía en caso de error crítico
        
    def get_current_search_dl_config(self) -> dict:
        """Obtiene la configuración actual de DL para búsqueda con manejo robusto de errores"""
        try:
            # Verificar disponibilidad de Deep Learning
            if not DEEP_LEARNING_AVAILABLE:
                return {}
                
            # Verificar que el checkbox existe y está habilitado
            if not hasattr(self, 'enable_dl_search_cb') or not self.enable_dl_search_cb.isChecked():
                return {}
            
            config = {'enabled': True, 'task_type': 'search'}
            
            # Obtener configuración con verificaciones
            try:
                if hasattr(self, 'dl_search_model_combo'):
                    model_text = self.dl_search_model_combo.currentText()
                    config['model_type'] = model_text.split(' - ')[0] if ' - ' in model_text else model_text
                else:
                    config['model_type'] = 'CNN'  # Valor por defecto
                    
                if hasattr(self, 'dl_search_confidence_spin'):
                    config['confidence_threshold'] = self.dl_search_confidence_spin.value()
                else:
                    config['confidence_threshold'] = 0.85  # Valor por defecto
                    
                if hasattr(self, 'dl_rerank_results_cb'):
                    config['rerank_results'] = self.dl_rerank_results_cb.isChecked()
                else:
                    config['rerank_results'] = False  # Valor por defecto
                    
            except Exception as e:
                print(f"Warning: Error obteniendo configuración de búsqueda DL: {e}")
                # Usar valores por defecto en caso de error
                config.update({
                    'model_type': 'CNN',
                    'confidence_threshold': 0.85,
                    'rerank_results': False
                })
                
            return config
            
        except Exception as e:
            print(f"Error crítico obteniendo configuración de búsqueda DL: {e}")
            return {}  # Retornar configuración vacía en caso de error crítico
        
    def on_comparison_model_configured(self, config: dict):
        """Maneja la configuración del modelo para comparación"""
        if 'model_type' in config:
            model_type = config['model_type']
            for i in range(self.dl_comparison_model_combo.count()):
                if self.dl_comparison_model_combo.itemText(i).startswith(model_type):
                    self.dl_comparison_model_combo.setCurrentIndex(i)
                    break
                    
        if 'confidence_threshold' in config:
            self.dl_confidence_spin.setValue(config['confidence_threshold'])
            
        # Guardar configuración avanzada
        self.advanced_comparison_dl_config = config
        
    def on_search_model_configured(self, config: dict):
        """Maneja la configuración del modelo para búsqueda"""
        if 'model_type' in config:
            model_type = config['model_type']
            for i in range(self.dl_search_model_combo.count()):
                if self.dl_search_model_combo.itemText(i).startswith(model_type):
                    self.dl_search_model_combo.setCurrentIndex(i)
                    break
                    
        if 'confidence_threshold' in config:
            self.dl_search_confidence_spin.setValue(config['confidence_threshold'])
            
        if 'rerank_results' in config:
            self.dl_rerank_results_cb.setChecked(config['rerank_results'])
            
        # Guardar configuración avanzada
        self.advanced_search_dl_config = config
        
        # Conexiones de resultados
        self.results_list.itemClicked.connect(self.on_result_selected)
        
    def on_mode_changed(self, index: int):
        """Maneja el cambio de modo"""
        self.current_mode = 'direct' if index == 0 else 'database'
        # Ya no necesitamos sincronizar con mode_combo porque fue removido
        
    def on_evidence_type_changed(self, evidence_type: str):
        """Maneja el cambio de tipo de evidencia"""
        # Actualizar opciones según el tipo de evidencia
        if "Bala" in evidence_type:
            self.analyze_striations_cb.setEnabled(True)
            self.analyze_land_groove_cb.setEnabled(True)
            self.analyze_firing_pin_cb.setEnabled(False)
            self.analyze_breech_face_cb.setEnabled(False)
        else:  # Casquillo
            self.analyze_striations_cb.setEnabled(False)
            self.analyze_land_groove_cb.setEnabled(False)
            self.analyze_firing_pin_cb.setEnabled(True)
            self.analyze_breech_face_cb.setEnabled(True)
            
    def update_cmc_threshold_label(self, value: int):
        """Actualiza la etiqueta del umbral CMC"""
        threshold = value / 100.0
        self.cmc_threshold_label.setText(f"{threshold:.2f}")
        
    def update_similarity_threshold_label(self, value: int):
        """Actualiza la etiqueta del umbral de similitud"""
        threshold = value / 100.0
        self.similarity_threshold_label.setText(f"{threshold:.2f}")
        
    def on_evidence_a_loaded(self, image_path: str):
        """Maneja la carga de la evidencia A"""
        self.comparison_data['evidence_a'] = image_path
        self.check_direct_ready()
        
    def on_evidence_b_loaded(self, image_path: str):
        """Maneja la carga de la evidencia B"""
        self.comparison_data['evidence_b'] = image_path
        self.check_direct_ready()
        
    def on_query_evidence_loaded(self, image_path: str):
        """Maneja la carga de la evidencia de consulta"""
        self.comparison_data['query_evidence'] = image_path
        # Verificar si query_image_viewer tiene el método load_image
        if hasattr(self.query_image_viewer, 'load_image'):
            self.query_image_viewer.load_image(image_path)
        else:
            # Fallback para QLabel
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.query_image_viewer.setPixmap(scaled_pixmap)
        self.search_button.setEnabled(True)
        self.db_step_indicator.set_current_step(1)
        # Habilitar los paneles de configuración de búsqueda cuando la evidencia esté cargada
        if hasattr(self, 'case_data_group'):
            self.case_data_group.setEnabled(True)
        if hasattr(self, 'processing_group'):
            self.processing_group.setEnabled(True)
        
    def check_direct_ready(self):
        """Verifica si la comparación directa está lista"""
        if ('evidence_a' in self.comparison_data and 
            'evidence_b' in self.comparison_data):
            self.analyze_button.setEnabled(True)
            self.direct_step_indicator.set_current_step(1)
            # Habilitar los paneles de configuración cuando ambas evidencias estén cargadas
            if hasattr(self, 'case_data_group'):
                self.case_data_group.setEnabled(True)
            if hasattr(self, 'processing_group'):
                self.processing_group.setEnabled(True)
            
    def start_ballistic_comparison(self):
        """Inicia la comparación balística directa"""
        if self.comparison_worker and self.comparison_worker.isRunning():
            return
            
        # Verificar que las imágenes estén cargadas
        if not (hasattr(self, 'evidence_a_zone') and self.evidence_a_zone.image_path):
            QMessageBox.warning(self, "Error", "Debe cargar la imagen de evidencia A")
            return
        if not (hasattr(self, 'evidence_b_zone') and self.evidence_b_zone.image_path):
            QMessageBox.warning(self, "Error", "Debe cargar la imagen de evidencia B")
            return
            
        evidence_type_text = self.evidence_type_combo.currentText()
        evidence_type = 'cartridge_case' if 'Casquillo' in evidence_type_text else 'bullet'
        
        comparison_params = {
            'mode': 'direct',
            'evidence_type': evidence_type,
            'evidence_a': self.evidence_a_zone.image_path,
            'evidence_b': self.evidence_b_zone.image_path,
            'cmc_threshold': self.cmc_threshold_slider.value() / 100.0,
            'case_data': {
                'case_number': self.case_number_edit.text(),
                'evidence_id_a': self.evidence_a_id_edit.text(),
                'evidence_id_b': self.evidence_b_id_edit.text(),
                'examiner': self.examiner_edit.text(),
                'weapon_make_a': self.weapon_a_make_edit.text(),
                'weapon_model_a': self.weapon_a_model_edit.text(),
                'weapon_caliber_a': self.caliber_a_edit.text(),
                'weapon_serial_a': getattr(self, 'weapon_a_serial_edit', QLineEdit()).text(),
                'weapon_make_b': self.weapon_b_make_edit.text(),
                'weapon_model_b': self.weapon_b_model_edit.text(),
                'weapon_caliber_b': self.caliber_b_edit.text(),
                'weapon_serial_b': getattr(self, 'weapon_b_serial_edit', QLineEdit()).text(),
                'case_description': self.case_description_edit.toPlainText()
            },
            'nist_metadata': {
                'laboratory': getattr(self, 'lab_name_edit', QLineEdit()).text(),
                'capture_device': getattr(self, 'capture_device_edit', QLineEdit()).text(),
                'lighting': getattr(self, 'lighting_type_combo', QComboBox()).currentText(),
                'calibration_date': getattr(self, 'calibration_date_edit', QLineEdit()).text()
            },
            'ballistic_features': {
                'firing_pin': getattr(self, 'compare_firing_pin_cb', QCheckBox()).isChecked(),
                'breech_face': getattr(self, 'compare_breech_face_cb', QCheckBox()).isChecked(),
                'extractor': getattr(self, 'compare_extractor_cb', QCheckBox()).isChecked(),
                'striations': getattr(self, 'compare_striations_cb', QCheckBox()).isChecked(),
                'land_groove': getattr(self, 'compare_land_groove_cb', QCheckBox()).isChecked()
            },
            'nist_validation': {
                'quality': getattr(self, 'nist_quality_check_cb', QCheckBox()).isChecked(),
                'metadata': getattr(self, 'nist_metadata_cb', QCheckBox()).isChecked(),
                'chain_custody': getattr(self, 'nist_authenticity_cb', QCheckBox()).isChecked()
            },
            'analysis_config': {
                'level': getattr(self, 'analysis_level_combo', QComboBox()).currentText(),
                'priority': getattr(self, 'priority_combo', QComboBox()).currentText(),
                'afte_conclusions': getattr(self, 'generate_afte_cb', QCheckBox()).isChecked(),
                'image_enhancement': getattr(self, 'contrast_enhancement_cb', QCheckBox()).isChecked(),
                'noise_reduction': getattr(self, 'noise_reduction_cb', QCheckBox()).isChecked(),
                'contrast_adjustment': getattr(self, 'contrast_enhancement_cb', QCheckBox()).isChecked()
            }
        }
        
        self.comparison_worker = BallisticComparisonWorker(comparison_params)
        self.comparison_worker.progressUpdated.connect(self.on_comparison_progress)
        self.comparison_worker.comparisonCompleted.connect(self.on_comparison_completed)
        self.comparison_worker.comparisonError.connect(self.on_comparison_error)
        
        self.direct_progress_card.show()
        self.direct_step_indicator.set_current_step(4)  # Updated to reflect new step count
        self.comparison_worker.start()
        
    def start_ballistic_search(self):
        """Inicia la búsqueda en base de datos balística"""
        if self.comparison_worker and self.comparison_worker.isRunning():
            return
            
        evidence_type_text = self.evidence_type_combo.currentText()
        evidence_type = 'cartridge_case' if 'Casquillo' in evidence_type_text else 'bullet'
        
        search_params = {
            'mode': 'database',
            'evidence_type': evidence_type,
            'query_image': self.comparison_data.get('query_evidence'),
            'filters': {
                'caliber': self.caliber_filter_combo.currentText(),
                'weapon_type': self.weapon_type_filter_combo.currentText(),
                'similarity_threshold': self.similarity_threshold_slider.value() / 100.0,
                'max_results': self.max_results_spinbox.value(),
                'date_from': self.date_from_edit.text(),
                'date_to': self.date_to_edit.text(),
                'case_number': self.case_number_filter_edit.text(),
                'location': self.location_filter_edit.text()
            }
        }
        
        self.comparison_worker = BallisticComparisonWorker(search_params)
        self.comparison_worker.progressUpdated.connect(self.on_search_progress)
        self.comparison_worker.comparisonCompleted.connect(self.on_search_completed)
        self.comparison_worker.comparisonError.connect(self.on_search_error)
        
        self.search_progress_card.show()
        self.db_step_indicator.set_current_step(2)
        self.comparison_worker.start()
        
    def on_comparison_progress(self, progress: int, message: str):
        """Actualiza el progreso de la comparación"""
        self.direct_progress_card.set_progress(progress, message)
        
    def on_search_progress(self, progress: int, message: str):
        """Actualiza el progreso de la búsqueda"""
        self.search_progress_card.set_progress(progress, message)
        
    def on_comparison_completed(self, results: dict):
        """Maneja la finalización de la comparación"""
        self.direct_progress_card.hide()
        self.direct_step_indicator.set_current_step(3)
        
        if results['mode'] == 'direct':
            self.display_ballistic_comparison_results(results)
        else:
            self.display_ballistic_search_results(results)
            
    def on_search_completed(self, results: dict):
        """Maneja la finalización de la búsqueda"""
        self.search_progress_card.hide()
        self.db_step_indicator.set_current_step(3)
        self.display_ballistic_search_results(results)
        
    def on_comparison_error(self, error_message: str):
        """Maneja errores en la comparación"""
        self.direct_progress_card.hide()
        QMessageBox.critical(self, "Error en Análisis Balístico", 
                           f"Error durante el análisis: {error_message}")
        
    def on_search_error(self, error_message: str):
        """Maneja errores en la búsqueda"""
        self.search_progress_card.hide()
        QMessageBox.critical(self, "Error en Búsqueda Balística", 
                           f"Error durante la búsqueda: {error_message}")
        
    def display_ballistic_comparison_results(self, results: dict):
        """Muestra los resultados de comparación balística con interfaz de pestañas"""
        # Crear widget de pestañas para resultados detallados
        results_tabs = QTabWidget()
        
        # Pestaña 1: Resumen
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        # Actualizar visualización CMC
        if hasattr(self, 'interactive_cmc') and self.interactive_cmc:
            self.interactive_cmc.update_cmc_data(results.get('cmc_analysis', {}))
        
        # Métricas principales
        metrics_group = QGroupBox("Métricas de Comparación")
        metrics_layout = QGridLayout(metrics_group)
        
        cmc_data = results.get('cmc_analysis', {})
        
        # Verificar que los labels existen antes de actualizarlos
        if hasattr(self, 'cmc_score_label') and self.cmc_score_label:
            self.cmc_score_label.setText(f"Score CMC: {results.get('cmc_score', 0):.3f}")
        if hasattr(self, 'total_cells_label') and self.total_cells_label:
            self.total_cells_label.setText(f"Células Totales: {cmc_data.get('total_cells', 0)}")
        if hasattr(self, 'valid_cells_label') and self.valid_cells_label:
            self.valid_cells_label.setText(f"Células Válidas: {cmc_data.get('valid_cells', 0)}")
        if hasattr(self, 'congruent_cells_label') and self.congruent_cells_label:
            self.congruent_cells_label.setText(f"Células Congruentes: {cmc_data.get('congruent_cells', 0)}")
        
        # Crear nuevos labels si no existen o fueron eliminados
        cmc_score_label = QLabel(f"Score CMC: {results.get('cmc_score', 0):.3f}")
        total_cells_label = QLabel(f"Células Totales: {cmc_data.get('total_cells', 0)}")
        valid_cells_label = QLabel(f"Células Válidas: {cmc_data.get('valid_cells', 0)}")
        congruent_cells_label = QLabel(f"Células Congruentes: {cmc_data.get('congruent_cells', 0)}")
        
        metrics_layout.addWidget(QLabel("Score CMC:"), 0, 0)
        metrics_layout.addWidget(cmc_score_label, 0, 1)
        metrics_layout.addWidget(QLabel("Células Totales:"), 1, 0)
        metrics_layout.addWidget(total_cells_label, 1, 1)
        metrics_layout.addWidget(QLabel("Células Válidas:"), 2, 0)
        metrics_layout.addWidget(valid_cells_label, 2, 1)
        metrics_layout.addWidget(QLabel("Células Congruentes:"), 3, 0)
        metrics_layout.addWidget(congruent_cells_label, 3, 1)
        
        summary_layout.addWidget(metrics_group)
        
        # Conclusión AFTE
        afte_conclusion = results.get('afte_conclusion', 'Inconclusive')
        result_type = results.get('result_type', 'warning')
        
        conclusion_text = f"Conclusión AFTE: {afte_conclusion}"
        if afte_conclusion == "Identification":
            conclusion_text += "\n✅ Las evidencias provienen del mismo arma de fuego"
        elif afte_conclusion == "Elimination":
            conclusion_text += "\n❌ Las evidencias NO provienen del mismo arma de fuego"
        else:
            conclusion_text += "\n⚠️ No se puede determinar con certeza el origen común"
            
        if hasattr(self, 'afte_conclusion_card') and self.afte_conclusion_card:
            self.afte_conclusion_card.set_value(conclusion_text, result_type)
            summary_layout.addWidget(self.afte_conclusion_card)
        
        results_tabs.addTab(summary_widget, "Resumen")
        
        # Pestaña 2: Análisis CMC Detallado
        cmc_widget = QWidget()
        cmc_layout = QVBoxLayout(cmc_widget)
        cmc_layout.addWidget(self.cmc_visualization)
        
        # Estadísticas detalladas
        stats_group = QGroupBox("Análisis Estadístico")
        stats_layout = QGridLayout(stats_group)
        
        stats = results.get('statistical_analysis', {})
        
        # Crear nuevos labels para estadísticas
        p_value_label = QLabel(f"{stats.get('p_value', 0):.4f}")
        ci = stats.get('confidence_interval', [0, 0])
        confidence_interval_label = QLabel(f"[{ci[0]:.3f}, {ci[1]:.3f}]")
        false_positive_rate_label = QLabel(f"{stats.get('false_positive_rate', 0):.4f}")
        
        stats_layout.addWidget(QLabel("Valor P:"), 0, 0)
        stats_layout.addWidget(p_value_label, 0, 1)
        stats_layout.addWidget(QLabel("Intervalo de Confianza:"), 1, 0)
        stats_layout.addWidget(confidence_interval_label, 1, 1)
        stats_layout.addWidget(QLabel("Tasa de Falsos Positivos:"), 2, 0)
        stats_layout.addWidget(false_positive_rate_label, 2, 1)
        
        cmc_layout.addWidget(stats_group)
        results_tabs.addTab(cmc_widget, "Análisis CMC")
        
        # Pestaña 3: Validación NIST
        if NIST_AVAILABLE:
            nist_widget = self.create_nist_validation_tab(results)
            results_tabs.addTab(nist_widget, "Validación NIST")
        
        # Pestaña 4: Características Balísticas
        features_widget = QWidget()
        features_layout = QVBoxLayout(features_widget)
        
        features_text = self.format_ballistic_features(results.get('ballistic_features', {}))
        self.ballistic_features_text.setText(features_text)
        features_layout.addWidget(self.ballistic_features_text)
        
        results_tabs.addTab(features_widget, "Características")
        
        # Reemplazar el contenido del panel de resultados
        if hasattr(self, 'direct_results_layout'):
            # Limpiar layout existente
            for i in reversed(range(self.direct_results_layout.count())):
                self.direct_results_layout.itemAt(i).widget().setParent(None)
            
            # Agregar pestañas de resultados
            self.direct_results_layout.addWidget(results_tabs)
            
            # Botones de acción
            actions_layout = QHBoxLayout()
            
            save_btn = QPushButton(" Guardar Resultados")
            save_btn.clicked.connect(lambda: self.save_comparison_results(results))
            
            report_btn = QPushButton(" Generar Reporte")
            report_btn.clicked.connect(lambda: self.generate_comparison_report(results))
            
            export_btn = QPushButton(" Exportar Datos")
            export_btn.clicked.connect(lambda: self.export_comparison_data(results))
            
            compare_db_btn = QPushButton(" Comparar con BD")
            compare_db_btn.clicked.connect(lambda: self.compare_with_database(results))
            
            actions_layout.addWidget(save_btn)
            actions_layout.addWidget(report_btn)
            actions_layout.addWidget(export_btn)
            actions_layout.addWidget(compare_db_btn)
            actions_layout.addStretch()
            
            self.direct_results_layout.addLayout(actions_layout)
        
        # Emitir señal de finalización
        self.comparisonCompleted.emit(results)
        
    def display_ballistic_search_results(self, results: dict):
        """Muestra los resultados de búsqueda balística"""
        # Actualizar estadísticas de búsqueda
        stats_text = (f"Búsqueda completada: {results.get('total_searched', 0)} evidencias analizadas, "
                     f"{results.get('candidates_found', 0)} candidatos encontrados, "
                     f"{results.get('high_confidence_matches', 0)} coincidencias de alta confianza")
        self.search_stats_label.setText(stats_text)
        
        # Limpiar y llenar lista de resultados
        self.results_list.clear()
        
        for result in results.get('results', []):
            item_widget = self.create_ballistic_result_item_widget(result)
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            item.setData(Qt.UserRole, result)
            
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, item_widget)
            
    def create_ballistic_result_item_widget(self, result: dict) -> QWidget:
        """Crea un widget para mostrar un resultado de búsqueda balística"""
        widget = QFrame()
        widget.setProperty("class", "result-item")
        
        layout = QHBoxLayout(widget)
        layout.setSpacing(15)
        
        # Información principal
        info_layout = QVBoxLayout()
        
        # Línea 1: ID y Score CMC
        header_layout = QHBoxLayout()
        
        id_label = QLabel(f"ID: {result.get('id', 'N/A')}")
        id_label.setProperty("class", "body-bold")
        header_layout.addWidget(id_label)
        
        header_layout.addStretch()
        
        cmc_score = result.get('cmc_score', 0)
        cmc_label = QLabel(f"CMC: {cmc_score:.3f}")
        cmc_label.setProperty("class", "body-bold")
        
        # Color según score CMC
        if cmc_score >= 0.85:
            cmc_label.setStyleSheet("color: #28a745; font-weight: bold;")
        elif cmc_score >= 0.70:
            cmc_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        else:
            cmc_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            
        header_layout.addWidget(cmc_label)
        
        info_layout.addLayout(header_layout)
        
        # Línea 2: Conclusión AFTE
        afte_label = QLabel(f"AFTE: {result.get('afte_conclusion', 'N/A')}")
        afte_label.setProperty("class", "body")
        info_layout.addWidget(afte_label)
        
        # Línea 3: Información del caso
        case_info = f"Caso: {result.get('case_number', 'N/A')} | {result.get('weapon_type', 'N/A')}"
        case_label = QLabel(case_info)
        case_label.setProperty("class", "caption")
        info_layout.addWidget(case_label)
        
        # Línea 4: Metadatos balísticos
        metadata = result.get('metadata', {})
        metadata_info = f"Calibre: {metadata.get('caliber', 'N/A')} | Fabricante: {metadata.get('manufacturer', 'N/A')}"
        metadata_label = QLabel(metadata_info)
        metadata_label.setProperty("class", "caption")
        info_layout.addWidget(metadata_label)
        
        layout.addLayout(info_layout)
        
        # Indicador visual de confianza
        confidence_indicator = QLabel("●")
        confidence_indicator.setProperty("class", "title")
        
        afte_conclusion = result.get('afte_conclusion', '')
        if afte_conclusion == 'Identification':
            confidence_indicator.setStyleSheet("color: #28a745; font-size: 20px;")
        elif afte_conclusion == 'Inconclusive':
            confidence_indicator.setStyleSheet("color: #ffc107; font-size: 20px;")
        else:
            confidence_indicator.setStyleSheet("color: #dc3545; font-size: 20px;")
            
        layout.addWidget(confidence_indicator)
        
        return widget
        
    def on_result_selected(self, item: QListWidgetItem):
        """Maneja la selección de un resultado de búsqueda"""
        result_data = item.data(Qt.UserRole)
        if result_data:
            self.selected_db_result = result_data
            
            # Cargar imagen seleccionada
            image_path = result_data.get('path', '')
            if image_path and os.path.exists(image_path):
                self.selected_image_viewer.load_image(image_path)
            
            # Actualizar métricas de comparación
            self.selected_cmc_score_label.setText(f"Score CMC: {result_data.get('cmc_score', 0):.3f}")
            self.selected_afte_conclusion_label.setText(f"Conclusión AFTE: {result_data.get('afte_conclusion', 'N/A')}")
            
            # Calcular confianza basada en CMC score
            cmc_score = result_data.get('cmc_score', 0)
            confidence = "Alta" if cmc_score >= 0.85 else "Media" if cmc_score >= 0.70 else "Baja"
            self.selected_confidence_label.setText(f"Confianza: {confidence}")
            
            case_info = f"Caso: {result_data.get('case_number', 'N/A')} | {result_data.get('weapon_type', 'N/A')}"
            self.selected_case_info_label.setText(case_info)
            
            self.db_step_indicator.set_current_step(3)
            
    def format_ballistic_features(self, features: dict) -> str:
        """Formatea las características balísticas para mostrar"""
        text_parts = []
        
        if 'firing_pin_correlation' in features:
            text_parts.append(f" Correlación Firing Pin: {features['firing_pin_correlation']:.3f}")
            
        if 'breech_face_correlation' in features:
            text_parts.append(f" Correlación Breech Face: {features['breech_face_correlation']:.3f}")
            
        if 'extractor_marks_correlation' in features:
            text_parts.append(f" Correlación Marcas Extractor: {features['extractor_marks_correlation']:.3f}")
            
        if features.get('striation_correlation'):
            text_parts.append(f" Correlación Estriado: {features['striation_correlation']:.3f}")
            
        return "\n".join(text_parts) if text_parts else "No hay características disponibles"
    
    def create_nist_validation_tab(self, results: dict) -> QWidget:
        """Crea la pestaña de validación NIST"""
        nist_widget = QWidget()
        nist_layout = QVBoxLayout(nist_widget)
        
        # Validación de calidad de imagen
        quality_group = QGroupBox("Validación de Calidad de Imagen")
        quality_layout = QVBoxLayout(quality_group)
        
        nist_data = results.get('nist_validation', {})
        quality_metrics = nist_data.get('quality_metrics', {})
        
        # Tabla de métricas de calidad
        quality_table = QTableWidget()
        quality_table.setColumnCount(3)
        quality_table.setHorizontalHeaderLabels(["Métrica", "Valor", "Estado"])
        quality_table.horizontalHeader().setStretchLastSection(True)
        
        metrics = [
            ("Resolución (DPI)", quality_metrics.get('resolution_dpi', 'N/A'), 
             "✅ Cumple" if quality_metrics.get('resolution_compliant', False) else "❌ No cumple"),
            ("Contraste", f"{quality_metrics.get('contrast', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('contrast_compliant', False) else "❌ No cumple"),
            ("Nitidez", f"{quality_metrics.get('sharpness', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('sharpness_compliant', False) else "❌ No cumple"),
            ("Ruido", f"{quality_metrics.get('noise_level', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('noise_compliant', False) else "❌ No cumple"),
            ("Iluminación", f"{quality_metrics.get('illumination_uniformity', 0):.3f}", 
             "✅ Cumple" if quality_metrics.get('illumination_compliant', False) else "❌ No cumple")
        ]
        
        quality_table.setRowCount(len(metrics))
        for i, (metric, value, status) in enumerate(metrics):
            quality_table.setItem(i, 0, QTableWidgetItem(metric))
            quality_table.setItem(i, 1, QTableWidgetItem(str(value)))
            quality_table.setItem(i, 2, QTableWidgetItem(status))
        
        quality_layout.addWidget(quality_table)
        nist_layout.addWidget(quality_group)
        
        # Validación de metadatos
        metadata_group = QGroupBox("Validación de Metadatos")
        metadata_layout = QVBoxLayout(metadata_group)
        
        metadata_table = QTableWidget()
        metadata_table.setColumnCount(2)
        metadata_table.setHorizontalHeaderLabels(["Campo", "Estado"])
        metadata_table.horizontalHeader().setStretchLastSection(True)
        
        metadata_validation = nist_data.get('metadata_validation', {})
        metadata_items = [
            ("Fecha de Captura", "✅ Presente" if metadata_validation.get('capture_date', False) else "❌ Faltante"),
            ("Información del Dispositivo", "✅ Presente" if metadata_validation.get('device_info', False) else "❌ Faltante"),
            ("Configuración de Cámara", "✅ Presente" if metadata_validation.get('camera_settings', False) else "❌ Faltante"),
            ("Cadena de Custodia", "✅ Válida" if metadata_validation.get('chain_of_custody', False) else "❌ Inválida"),
            ("Hash de Integridad", "✅ Verificado" if metadata_validation.get('integrity_hash', False) else "❌ No verificado")
        ]
        
        metadata_table.setRowCount(len(metadata_items))
        for i, (field, status) in enumerate(metadata_items):
            metadata_table.setItem(i, 0, QTableWidgetItem(field))
            metadata_table.setItem(i, 1, QTableWidgetItem(status))
        
        metadata_layout.addWidget(metadata_table)
        nist_layout.addWidget(metadata_group)
        
        # Reporte de cumplimiento general
        compliance_group = QGroupBox("Cumplimiento General NIST")
        compliance_layout = QVBoxLayout(compliance_group)
        
        overall_compliance = nist_data.get('overall_compliance', False)
        compliance_score = nist_data.get('compliance_score', 0)
        
        compliance_label = QLabel()
        if overall_compliance:
            compliance_label.setText(f"✅ CUMPLE con estándares NIST (Puntuación: {compliance_score:.1f}/100)")
            compliance_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            compliance_label.setText(f"❌ NO CUMPLE con estándares NIST (Puntuación: {compliance_score:.1f}/100)")
            compliance_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        
        compliance_layout.addWidget(compliance_label)
        
        # Recomendaciones
        recommendations = nist_data.get('recommendations', [])
        if recommendations:
            rec_label = QLabel("Recomendaciones:")
            rec_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
            compliance_layout.addWidget(rec_label)
            
            for rec in recommendations:
                rec_item = QLabel(f"• {rec}")
                rec_item.setWordWrap(True)
                compliance_layout.addWidget(rec_item)
        
        nist_layout.addWidget(compliance_group)
        
        return nist_widget
    
    def save_comparison_results(self, results: dict):
        """Guarda los resultados de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar Resultados de Comparación", 
                f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
                QMessageBox.information(self, "Éxito", 
                                      f"Resultados guardados exitosamente en:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar resultados:\n{str(e)}")
    
    def generate_comparison_report(self, results: dict):
        """Genera un reporte de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generar Reporte de Comparación", 
                f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "PDF Files (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Aquí se integraría con el módulo de reportes
                QMessageBox.information(self, "Reporte", 
                                      f"Funcionalidad de reporte será implementada.\nArchivo: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar reporte:\n{str(e)}")
    
    def export_comparison_data(self, results: dict):
        """Exporta los datos de comparación"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar Datos de Comparación", 
                f"comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                # Aquí se implementaría la exportación a CSV/Excel
                QMessageBox.information(self, "Exportación", 
                                      f"Funcionalidad de exportación será implementada.\nArchivo: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar datos:\n{str(e)}")
    
    def compare_with_database(self, results: dict):
        """Compara los resultados con la base de datos"""
        try:
            # Cambiar a la pestaña de búsqueda en base de datos
            self.mode_tabs.setCurrentIndex(1)  # Asumiendo que es la segunda pestaña
            
            QMessageBox.information(self, "Comparación con BD", 
                                  "Cambiando a modo de búsqueda en base de datos.\n"
                                  "Configure los parámetros y ejecute la búsqueda.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cambiar a búsqueda en BD:\n{str(e)}")
    
    def save_comparison_configuration(self):
        """Guarda la configuración de comparación directa"""
        try:
            config_data = {
                'mode': 'direct_comparison',
                'evidence_type': self.evidence_type_combo.currentText(),
                'cmc_threshold': self.cmc_threshold_slider.value(),
                'analysis_method': self.analysis_method_combo.currentText(),
                'afte_criteria': self.afte_criteria_combo.currentText(),
                'timestamp': datetime.now().isoformat()
            }
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Configuración de Comparación",
                f"comparison_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "Archivos JSON (*.json)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                QMessageBox.information(self, "Éxito", "Configuración de comparación guardada correctamente")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración: {str(e)}")
    
    def reset_comparison_configuration(self):
        """Reinicia la configuración de comparación directa"""
        reply = QMessageBox.question(
            self,
            "Confirmar Reinicio",
            "¿Está seguro de que desea reiniciar la configuración de comparación?\n\nSe perderán todos los ajustes actuales.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reiniciar controles
            self.evidence_type_combo.setCurrentIndex(0)
            self.cmc_threshold_slider.setValue(70)
            self.analysis_method_combo.setCurrentIndex(0)
            self.afte_criteria_combo.setCurrentIndex(0)
            
            # Limpiar zonas de imagen
            self.evidence_a_zone.clear()
            self.evidence_b_zone.clear()
            
            # Deshabilitar botón de análisis
            self.analyze_button.setEnabled(False)
            
            QMessageBox.information(self, "Configuración Reiniciada", "La configuración de comparación ha sido reiniciada")
    
    def save_search_configuration(self):
        """Guarda la configuración de búsqueda en base de datos"""
        try:
            config_data = {
                'mode': 'database_search',
                'evidence_type': self.evidence_type_combo.currentText(),
                'similarity_threshold': self.similarity_threshold_slider.value(),
                'max_results': self.max_results_spin.value(),
                'timestamp': datetime.now().isoformat()
            }
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Configuración de Búsqueda",
                f"search_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "Archivos JSON (*.json)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                QMessageBox.information(self, "Éxito", "Configuración de búsqueda guardada correctamente")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración: {str(e)}")
    
    def reset_search_configuration(self):
        """Reinicia la configuración de búsqueda en base de datos"""
        reply = QMessageBox.question(
            self,
            "Confirmar Reinicio",
            "¿Está seguro de que desea reiniciar la configuración de búsqueda?\n\nSe perderán todos los ajustes actuales.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reiniciar controles
            self.evidence_type_combo.setCurrentIndex(0)
            self.similarity_threshold_slider.setValue(80)
            self.max_results_spin.setValue(10)
            
            # Limpiar zona de imagen
            self.query_evidence_zone.clear()
            
            # Deshabilitar botón de búsqueda
            self.search_button.setEnabled(False)
            
            QMessageBox.information(self, "Configuración Reiniciada", "La configuración de búsqueda ha sido reiniciada")
    
    def next_step(self):
        """Avanza al siguiente paso del flujo de trabajo"""
        if self.current_step < 5:  # Máximo 5 pasos (0-4)
            self.current_step += 1
            self.update_step_visibility()
            self.update_navigation_buttons()
    
    def prev_step(self):
        """Retrocede al paso anterior del flujo de trabajo"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step_visibility()
            self.update_navigation_buttons()
    
    def update_step_visibility(self):
        """Actualiza la visibilidad y habilitación de los pasos según el paso actual"""
        # Paso 1: Selección de evidencia (siempre habilitado)
        if hasattr(self, 'evidence_selection_group'):
            self.evidence_selection_group.setEnabled(True)
        
        # Paso 2: Datos del caso (habilitado desde paso 1)
        if hasattr(self, 'case_data_group'):
            self.case_data_group.setEnabled(self.current_step >= 1)
        
        # Paso 3: Metadatos NIST (habilitado desde paso 2)
        if hasattr(self, 'nist_group'):
            self.nist_group.setEnabled(self.current_step >= 2)
        
        # Paso 4: Configuración de análisis (habilitado desde paso 3)
        if hasattr(self, 'processing_group'):
            self.processing_group.setEnabled(self.current_step >= 3)
        
        # Paso 5: Configuración CMC y AFTE (habilitado desde paso 4)
        if hasattr(self, 'ballistic_config_group'):
            self.ballistic_config_group.setEnabled(self.current_step >= 4)
    
    def update_navigation_buttons(self):
        """Actualiza el estado de los botones de navegación"""
        # Botón anterior: habilitado si no estamos en el primer paso
        self.prev_button.setEnabled(self.current_step > 0)
        
        # Botón siguiente: habilitado según el paso actual y validaciones
        if self.current_step == 0:
            # Paso 0: Verificar que hay imágenes cargadas
            if self.current_mode == 0:  # Modo directo
                self.next_button.setEnabled(
                    hasattr(self, 'evidence_a_zone') and self.evidence_a_zone.image_path and
                    hasattr(self, 'evidence_b_zone') and self.evidence_b_zone.image_path
                )
            else:  # Modo búsqueda
                self.next_button.setEnabled(
                    hasattr(self, 'query_evidence_zone') and self.query_evidence_zone.image_path
                )
        elif self.current_step < 4:
            # Pasos intermedios: siempre habilitado para avanzar
            self.next_button.setEnabled(True)
        else:
            # Último paso
            self.next_button.setEnabled(False)
    
    def reset_workflow(self):
        """Reinicia el flujo de trabajo completo"""
        reply = QMessageBox.question(
            self, 
            "Reiniciar Flujo de Trabajo",
            "¿Está seguro de que desea reiniciar todo el flujo de trabajo?\n"
            "Se perderán todos los datos y configuraciones actuales.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reiniciar paso actual
            self.current_step = 0
            
            # Limpiar zonas de imagen
            if hasattr(self, 'evidence_a_zone'):
                self.evidence_a_zone.clear()
            if hasattr(self, 'evidence_b_zone'):
                self.evidence_b_zone.clear()
            if hasattr(self, 'query_evidence_zone'):
                self.query_evidence_zone.clear()
            
            # Reiniciar configuraciones
            self.reset_comparison_configuration()
            self.reset_search_configuration()
            
            # Limpiar resultados
            if hasattr(self, 'comparison_data'):
                self.comparison_data = None
            
            # Actualizar botones de navegación
            self.update_navigation_buttons()
            
            QMessageBox.information(self, "Flujo Reiniciado", "El flujo de trabajo ha sido reiniciado completamente")