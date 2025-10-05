#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Integration Module - SIGeC-BalisticaGUI
==========================================

Módulo de integración que conecta la GUI moderna con el backend existente
del sistema SIGeC-Balistica, proporcionando una interfaz unificada para:

- Análisis estadístico unificado
- Integración NIST
- Procesamiento de imágenes
- Matching y comparación
- Configuración unificada
- Base de datos

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QApplication

# Importar configuración unificada
try:
    from config.unified_config import (
        UnifiedConfig, get_unified_config,
        get_database_config, get_image_processing_config,
        get_matching_config, get_gui_config, get_nist_config
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Configuración unificada no disponible: {e}")
    CONFIG_AVAILABLE = False

# Importar análisis estadístico
try:
    from common.statistical_core import UnifiedStatisticalAnalysis
    STATISTICAL_CORE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Núcleo estadístico no disponible: {e}")
    STATISTICAL_CORE_AVAILABLE = False

# Importar integración NIST
try:
    from common.nist_integration import NISTStatisticalIntegration
    NIST_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Integración NIST no disponible: {e}")
    NIST_INTEGRATION_AVAILABLE = False

# Importar procesamiento de imágenes
try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Procesamiento de imágenes no disponible: {e}")
    IMAGE_PROCESSING_AVAILABLE = False

# Importar matching
try:
    from matching.unified_matcher import UnifiedMatcher, MatchingLevel, AlgorithmType
    MATCHING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sistema de matching no disponible: {e}")
    MATCHING_AVAILABLE = False

# Importar base de datos
try:
    from database.unified_database import UnifiedDatabase
    DATABASE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Base de datos no disponible: {e}")
    DATABASE_AVAILABLE = False

# Configurar logging
logger = logging.getLogger(__name__)

class AnalysisStatus(Enum):
    """Estados del análisis"""
    IDLE = "idle"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class ProcessingMode(Enum):
    """Modos de procesamiento"""
    INDIVIDUAL = "individual"
    COMPARISON_DIRECT = "comparison_direct"
    COMPARISON_DATABASE = "comparison_database"

@dataclass
class AnalysisResult:
    """Resultado de análisis unificado"""
    status: AnalysisStatus
    mode: ProcessingMode
    processing_time: float
    
    # Datos de entrada
    image_path: Optional[str] = None
    image_data: Optional[np.ndarray] = None
    case_data: Optional[Dict[str, Any]] = None
    nist_metadata: Optional[Dict[str, Any]] = None
    
    # Resultados de procesamiento
    processed_image: Optional[np.ndarray] = None
    features: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    
    # Resultados estadísticos
    statistical_results: Optional[Dict[str, Any]] = None
    nist_compliance: Optional[Dict[str, Any]] = None
    
    # Resultados de comparación
    similarity_score: Optional[float] = None
    matches: Optional[List[Dict[str, Any]]] = None
    comparison_results: Optional[Dict[str, Any]] = None
    
    # Visualizaciones
    visualizations: Optional[Dict[str, Any]] = None
    
    # Errores
    error_message: Optional[str] = None
    error_details: Optional[str] = None

class BackendIntegration(QObject):
    """
    Clase principal de integración con el backend
    
    Proporciona una interfaz unificada para acceder a todas las
    funcionalidades del backend desde la GUI.
    """
    
    # Señales para comunicación con la GUI
    status_changed = pyqtSignal(AnalysisStatus)
    progress_updated = pyqtSignal(int, str)  # porcentaje, mensaje
    analysis_completed = pyqtSignal(AnalysisResult)
    error_occurred = pyqtSignal(str, str)  # mensaje, detalles
    
    def __init__(self):
        super().__init__()
        
        # Estado interno
        self.current_status = AnalysisStatus.IDLE
        self.current_analysis = None
        
        # Componentes del backend
        self.config = None
        self.statistical_analyzer = None
        self.nist_integration = None
        self.image_processor = None
        self.matcher = None
        self.database = None
        
        # Inicializar componentes
        self._initialize_backend_components()
        
        logger.info("BackendIntegration inicializado correctamente")
    
    def _initialize_backend_components(self):
        """Inicializa todos los componentes del backend"""
        try:
            # Configuración unificada
            if CONFIG_AVAILABLE:
                self.config = get_unified_config()
                logger.info("Configuración unificada cargada")
            
            # Análisis estadístico
            if STATISTICAL_CORE_AVAILABLE:
                self.statistical_analyzer = UnifiedStatisticalAnalysis()
                logger.info("Analizador estadístico inicializado")
            
            # Integración NIST
            if NIST_INTEGRATION_AVAILABLE:
                self.nist_integration = NISTStatisticalIntegration()
                logger.info("Integración NIST inicializada")
            
            # Procesamiento de imágenes
            if IMAGE_PROCESSING_AVAILABLE and self.config:
                img_config = get_image_processing_config()
                self.image_processor = UnifiedPreprocessor(img_config)
                logger.info("Procesador de imágenes inicializado")
            
            # Sistema de matching
            if MATCHING_AVAILABLE and self.config:
                match_config = get_matching_config()
                self.matcher = UnifiedMatcher(match_config)
                logger.info("Sistema de matching inicializado")
            
            # Base de datos
            if DATABASE_AVAILABLE and self.config:
                # Pasar la configuración completa en lugar de solo la sección de base de datos
                self.database = UnifiedDatabase(self.config)
                logger.info("Base de datos inicializada")
                
        except Exception as e:
            logger.error(f"Error inicializando componentes del backend: {e}")
            logger.error(traceback.format_exc())
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema"""
        return {
            'config_available': CONFIG_AVAILABLE and self.config is not None,
            'statistical_available': STATISTICAL_CORE_AVAILABLE and self.statistical_analyzer is not None,
            'nist_available': NIST_INTEGRATION_AVAILABLE and self.nist_integration is not None,
            'image_processing_available': IMAGE_PROCESSING_AVAILABLE and self.image_processor is not None,
            'matching_available': MATCHING_AVAILABLE and self.matcher is not None,
            'database_available': DATABASE_AVAILABLE and self.database is not None,
            'current_status': self.current_status.value,
            'backend_version': '1.0.0'
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Obtiene la configuración actual del sistema"""
        if not self.config:
            return {}
        
        return {
            'database': get_database_config().__dict__ if CONFIG_AVAILABLE else {},
            'image_processing': get_image_processing_config().__dict__ if CONFIG_AVAILABLE else {},
            'matching': get_matching_config().__dict__ if CONFIG_AVAILABLE else {},
            'gui': get_gui_config().__dict__ if CONFIG_AVAILABLE else {},
            'nist': get_nist_config().__dict__ if CONFIG_AVAILABLE else {}
        }
    
    def update_configuration(self, section: str, **kwargs) -> bool:
        """Actualiza la configuración del sistema"""
        if not self.config:
            return False
        
        try:
            self.config.update_config(section, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")
            return False
    
    def validate_image(self, image_path: str) -> Tuple[bool, str]:
        """Valida si una imagen es procesable"""
        if not os.path.exists(image_path):
            return False, "El archivo no existe"
        
        if not self.image_processor:
            return False, "Procesador de imágenes no disponible"
        
        try:
            # Intentar cargar la imagen
            image = self.image_processor.load_image(image_path)
            if image is None:
                return False, "No se pudo cargar la imagen"
            
            # Validar formato y tamaño
            if self.config:
                img_config = get_image_processing_config()
                height, width = image.shape[:2]
                
                if width < img_config.min_image_size or height < img_config.min_image_size:
                    return False, f"Imagen demasiado pequeña (mínimo: {img_config.min_image_size}px)"
                
                if width > img_config.max_image_size or height > img_config.max_image_size:
                    return False, f"Imagen demasiado grande (máximo: {img_config.max_image_size}px)"
            
            return True, "Imagen válida"
            
        except Exception as e:
            return False, f"Error validando imagen: {str(e)}"
    
    def get_supported_formats(self) -> List[str]:
        """Obtiene los formatos de imagen soportados"""
        if self.config:
            img_config = get_image_processing_config()
            return img_config.supported_formats
        return ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    def get_processing_algorithms(self) -> Dict[str, List[str]]:
        """Obtiene los algoritmos de procesamiento disponibles"""
        algorithms = {
            'feature_extraction': [],
            'matching': [],
            'quality_assessment': []
        }
        
        if MATCHING_AVAILABLE:
            algorithms['feature_extraction'] = [alg.value for alg in AlgorithmType]
            algorithms['matching'] = ['BF', 'FLANN', 'HYBRID']
        
        if IMAGE_PROCESSING_AVAILABLE:
            algorithms['quality_assessment'] = ['NIST', 'Custom', 'Statistical']
        
        return algorithms
    
    def get_matching_levels(self) -> List[str]:
        """Obtiene los niveles de matching disponibles"""
        if MATCHING_AVAILABLE:
            return [level.value for level in MatchingLevel]
        return ['basic', 'standard', 'advanced']
    
    def search_database(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca en la base de datos"""
        if not self.database:
            return []
        
        try:
            return self.database.search(**query_params)
        except Exception as e:
            logger.error(f"Error buscando en base de datos: {e}")
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la base de datos"""
        if not self.database:
            return {}
        
        try:
            return self.database.get_statistics()
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def export_results(self, results: AnalysisResult, format: str = 'json') -> Optional[str]:
        """Exporta resultados de análisis"""
        try:
            if format.lower() == 'json':
                import json
                export_data = {
                    'status': results.status.value,
                    'mode': results.mode.value,
                    'processing_time': results.processing_time,
                    'case_data': results.case_data,
                    'nist_metadata': results.nist_metadata,
                    'quality_metrics': results.quality_metrics,
                    'statistical_results': results.statistical_results,
                    'nist_compliance': results.nist_compliance,
                    'similarity_score': results.similarity_score,
                    'comparison_results': results.comparison_results
                }
                
                # Crear directorio de salida si no existe
                output_dir = Path("output/exports")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generar nombre de archivo único
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_results_{timestamp}.json"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                return str(filepath)
                
        except Exception as e:
            logger.error(f"Error exportando resultados: {e}")
            return None
    
    def _update_status(self, status: AnalysisStatus):
        """Actualiza el estado interno y emite señal"""
        self.current_status = status
        self.status_changed.emit(status)
    
    def _emit_progress(self, percentage: int, message: str):
        """Emite señal de progreso"""
        self.progress_updated.emit(percentage, message)
    
    def _emit_error(self, message: str, details: str = ""):
        """Emite señal de error"""
        self.error_occurred.emit(message, details)
        self._update_status(AnalysisStatus.ERROR)

# Instancia global para acceso desde toda la GUI
_backend_integration = None

def get_backend_integration() -> BackendIntegration:
    """Obtiene la instancia global de integración con el backend"""
    global _backend_integration
    if _backend_integration is None:
        _backend_integration = BackendIntegration()
    return _backend_integration

def initialize_backend_integration() -> BackendIntegration:
    """Inicializa la integración con el backend"""
    global _backend_integration
    _backend_integration = BackendIntegration()
    return _backend_integration

def reset_backend_integration():
    """Resetea la integración con el backend"""
    global _backend_integration
    _backend_integration = None

if __name__ == "__main__":
    # Prueba básica del módulo
    logging.basicConfig(level=logging.INFO)
    
    print("=== Backend Integration Test ===")
    
    backend = BackendIntegration()
    status = backend.get_system_status()
    
    print("Estado del sistema:")
    for component, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}: {available}")
    
    print("\nFormatos soportados:", backend.get_supported_formats())
    print("Algoritmos disponibles:", backend.get_processing_algorithms())
    print("Niveles de matching:", backend.get_matching_levels())
    
    print("\n✅ Módulo de integración inicializado correctamente")