"""
Gestor de Estado de la Aplicación - GUI Unificada
Sistema Balístico Forense SIGeC-Balistica

Maneja el estado compartido entre pestañas y paneles acoplables,
integrando funcionalidades de ambas implementaciones (gui y gui 1)
"""

import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class ImageAnalysisResult:
    """Resultado de análisis de imagen individual"""
    image_path: str
    timestamp: datetime
    features_extracted: int
    roi_detected: bool
    quality_score: float
    processing_time: float
    metadata: Dict[str, Any]
    nist_compliant: bool = True


@dataclass
class ComparisonResult:
    """Resultado de comparación directa entre dos imágenes"""
    image1_path: str
    image2_path: str
    timestamp: datetime
    similarity_percentage: float
    matches_found: int
    algorithm_used: str
    processing_time: float
    match_details: Dict[str, Any]
    correlation_score: float = 0.0
    ssim_score: float = 0.0
    mse_score: float = 0.0


@dataclass
class SearchResult:
    """Resultado de búsqueda en base de datos"""
    query_image_path: str
    timestamp: datetime
    results_found: int
    top_matches: List[Dict[str, Any]]
    search_parameters: Dict[str, Any]
    processing_time: float


@dataclass
class QualityMetrics:
    """Métricas de calidad de imagen"""
    overall_quality: float
    focus: float
    noise: float
    exposure: float
    detail: float
    recommendations: str


class AppStateManager(QObject):
    """Gestor centralizado del estado de la aplicación GUI unificada"""
    
    # Señales para notificar cambios de estado
    image_analysis_updated = pyqtSignal(object)  # ImageAnalysisResult
    comparison_updated = pyqtSignal(object)      # ComparisonResult
    search_updated = pyqtSignal(object)          # SearchResult
    quality_metrics_updated = pyqtSignal(object) # QualityMetrics
    statistics_updated = pyqtSignal(dict)        # Dict con estadísticas
    metadata_updated = pyqtSignal(dict)          # Dict con metadatos NIST
    state_cleared = pyqtSignal()
    active_tab_changed = pyqtSignal(str)         # Nombre de la pestaña activa
    zoom_level_changed = pyqtSignal(float)       # Nivel de zoom
    quality_threshold_changed = pyqtSignal(float) # Umbral de calidad
    
    def __init__(self):
        super().__init__()
        self._state = {
            # Resultados de análisis
            'last_image_analysis': None,
            'last_comparison': None,
            'last_search': None,
            'last_quality_metrics': None,
            
            # Historial
            'analysis_history': [],
            'comparison_history': [],
            'search_history': [],
            
            # Estado de la interfaz
            'active_tab': 'análisis',
            'zoom_level': 1.0,
            'quality_threshold': 0.3,
            'show_keypoints': True,
            'show_matches': True,
            'sync_zoom': True,
            'top_matches_count': 50,
            
            # Estadísticas en tiempo real
            'current_statistics': {},
            
            # Metadatos NIST del caso actual
            'current_metadata': {},
            
            # Configuración de paneles
            'dock_visibility': {
                'visualization_controls': True,
                'statistics': True,
                'metadata': True,
                'quality': False
            }
        }
        
    def update_image_analysis(self, result: ImageAnalysisResult):
        """Actualizar resultado de análisis de imagen"""
        self._state['last_image_analysis'] = result
        self._state['analysis_history'].insert(0, result)
        
        # Mantener solo los últimos 10 análisis
        if len(self._state['analysis_history']) > 10:
            self._state['analysis_history'] = self._state['analysis_history'][:10]
            
        self.image_analysis_updated.emit(result)
        
    def update_comparison(self, result: ComparisonResult):
        """Actualizar resultado de comparación"""
        self._state['last_comparison'] = result
        self._state['comparison_history'].insert(0, result)
        
        # Mantener solo las últimas 10 comparaciones
        if len(self._state['comparison_history']) > 10:
            self._state['comparison_history'] = self._state['comparison_history'][:10]
            
        # Actualizar estadísticas automáticamente
        self.update_statistics({
            'correlation': result.correlation_score,
            'ssim': result.ssim_score,
            'mse': result.mse_score,
            'matches_count': result.matches_found
        })
            
        self.comparison_updated.emit(result)
        
    def update_search(self, result: SearchResult):
        """Actualizar resultado de búsqueda"""
        self._state['last_search'] = result
        self._state['search_history'].insert(0, result)
        
        # Mantener solo las últimas 10 búsquedas
        if len(self._state['search_history']) > 10:
            self._state['search_history'] = self._state['search_history'][:10]
            
        self.search_updated.emit(result)
        
    def update_quality_metrics(self, metrics: QualityMetrics):
        """Actualizar métricas de calidad"""
        self._state['last_quality_metrics'] = metrics
        self.quality_metrics_updated.emit(metrics)
        
    def update_statistics(self, stats_dict: Dict[str, Any]):
        """Actualizar estadísticas en tiempo real"""
        self._state['current_statistics'].update(stats_dict)
        self.statistics_updated.emit(self._state['current_statistics'])
        
    def update_metadata(self, metadata_dict: Dict[str, Any]):
        """Actualizar metadatos NIST"""
        self._state['current_metadata'].update(metadata_dict)
        self.metadata_updated.emit(self._state['current_metadata'])
        
    def set_active_tab(self, tab_name: str):
        """Establecer la pestaña activa"""
        self._state['active_tab'] = tab_name.lower()
        self.active_tab_changed.emit(tab_name)
        
    def set_zoom_level(self, zoom: float):
        """Establecer el nivel de zoom"""
        self._state['zoom_level'] = zoom
        self.zoom_level_changed.emit(zoom)
        
    def set_quality_threshold(self, threshold: float):
        """Establecer el umbral de calidad"""
        self._state['quality_threshold'] = threshold
        self.quality_threshold_changed.emit(threshold)
        
    def set_visualization_option(self, option: str, value: Any):
        """Establecer opciones de visualización"""
        if option in ['show_keypoints', 'show_matches', 'sync_zoom']:
            self._state[option] = value
        elif option == 'top_matches_count':
            self._state['top_matches_count'] = value
            
    def set_dock_visibility(self, dock_name: str, visible: bool):
        """Establecer visibilidad de panel acoplable"""
        self._state['dock_visibility'][dock_name] = visible
        
    # Getters
    def get_last_image_analysis(self) -> Optional[ImageAnalysisResult]:
        """Obtener último análisis de imagen"""
        return self._state['last_image_analysis']
        
    def get_last_comparison(self) -> Optional[ComparisonResult]:
        """Obtener última comparación"""
        return self._state['last_comparison']
        
    def get_last_search(self) -> Optional[SearchResult]:
        """Obtener última búsqueda"""
        return self._state['last_search']
        
    def get_last_quality_metrics(self) -> Optional[QualityMetrics]:
        """Obtener últimas métricas de calidad"""
        return self._state['last_quality_metrics']
        
    def get_current_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas actuales"""
        return self._state['current_statistics'].copy()
        
    def get_current_metadata(self) -> Dict[str, Any]:
        """Obtener metadatos actuales"""
        return self._state['current_metadata'].copy()
        
    def get_active_tab(self) -> str:
        """Obtener pestaña activa"""
        return self._state['active_tab']
        
    def get_zoom_level(self) -> float:
        """Obtener nivel de zoom"""
        return self._state['zoom_level']
        
    def get_quality_threshold(self) -> float:
        """Obtener umbral de calidad"""
        return self._state['quality_threshold']
        
    def get_visualization_options(self) -> Dict[str, Any]:
        """Obtener opciones de visualización"""
        return {
            'show_keypoints': self._state['show_keypoints'],
            'show_matches': self._state['show_matches'],
            'sync_zoom': self._state['sync_zoom'],
            'top_matches_count': self._state['top_matches_count']
        }
        
    def get_dock_visibility(self) -> Dict[str, bool]:
        """Obtener visibilidad de paneles acoplables"""
        return self._state['dock_visibility'].copy()
        
    def get_analysis_history(self) -> List[ImageAnalysisResult]:
        """Obtener historial de análisis"""
        return self._state['analysis_history'].copy()
        
    def get_comparison_history(self) -> List[ComparisonResult]:
        """Obtener historial de comparaciones"""
        return self._state['comparison_history'].copy()
        
    def get_search_history(self) -> List[SearchResult]:
        """Obtener historial de búsquedas"""
        return self._state['search_history'].copy()
        
    def has_recent_data(self) -> Dict[str, bool]:
        """Verificar si hay datos recientes disponibles"""
        return {
            'analysis': self._state['last_image_analysis'] is not None,
            'comparison': self._state['last_comparison'] is not None,
            'search': self._state['last_search'] is not None,
            'quality': self._state['last_quality_metrics'] is not None
        }
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Obtener resumen del estado actual"""
        return {
            'active_tab': self._state['active_tab'],
            'zoom_level': self._state['zoom_level'],
            'quality_threshold': self._state['quality_threshold'],
            'has_analysis': self._state['last_image_analysis'] is not None,
            'has_comparison': self._state['last_comparison'] is not None,
            'has_search': self._state['last_search'] is not None,
            'analysis_count': len(self._state['analysis_history']),
            'comparison_count': len(self._state['comparison_history']),
            'search_count': len(self._state['search_history']),
            'dock_visibility': self._state['dock_visibility'].copy(),
            'visualization_options': self.get_visualization_options()
        }
        
    def clear_state(self):
        """Limpiar todo el estado"""
        self._state.update({
            'last_image_analysis': None,
            'last_comparison': None,
            'last_search': None,
            'last_quality_metrics': None,
            'analysis_history': [],
            'comparison_history': [],
            'search_history': [],
            'current_statistics': {},
            'current_metadata': {}
        })
        self.state_cleared.emit()
        
    def export_state_to_json(self, file_path: str):
        """Exportar estado a archivo JSON"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'analysis_history': [self._serialize_analysis(a) for a in self._state['analysis_history']],
                'comparison_history': [self._serialize_comparison(c) for c in self._state['comparison_history']],
                'search_history': [self._serialize_search(s) for s in self._state['search_history']],
                'current_metadata': self._state['current_metadata'],
                'visualization_options': self.get_visualization_options()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error exportando estado: {e}")
            
    def import_state_from_json(self, file_path: str):
        """Importar estado desde archivo JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
                
            # Importar historiales
            self._state['analysis_history'] = [
                self._deserialize_analysis(a) for a in import_data.get('analysis_history', [])
            ]
            self._state['comparison_history'] = [
                self._deserialize_comparison(c) for c in import_data.get('comparison_history', [])
            ]
            self._state['search_history'] = [
                self._deserialize_search(s) for s in import_data.get('search_history', [])
            ]
            
            # Importar metadatos
            if 'current_metadata' in import_data:
                self.update_metadata(import_data['current_metadata'])
                
            # Importar opciones de visualización
            if 'visualization_options' in import_data:
                options = import_data['visualization_options']
                for key, value in options.items():
                    self.set_visualization_option(key, value)
                    
        except Exception as e:
            print(f"Error importando estado: {e}")
            
    def _serialize_analysis(self, analysis: ImageAnalysisResult) -> Dict[str, Any]:
        """Serializar análisis para JSON"""
        data = asdict(analysis)
        data['timestamp'] = analysis.timestamp.isoformat()
        return data
        
    def _serialize_comparison(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Serializar comparación para JSON"""
        data = asdict(comparison)
        data['timestamp'] = comparison.timestamp.isoformat()
        return data
        
    def _serialize_search(self, search: SearchResult) -> Dict[str, Any]:
        """Serializar búsqueda para JSON"""
        data = asdict(search)
        data['timestamp'] = search.timestamp.isoformat()
        return data
        
    def _deserialize_analysis(self, data: Dict[str, Any]) -> ImageAnalysisResult:
        """Deserializar análisis desde JSON"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return ImageAnalysisResult(**data)
        
    def _deserialize_comparison(self, data: Dict[str, Any]) -> ComparisonResult:
        """Deserializar comparación desde JSON"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return ComparisonResult(**data)
        
    def _deserialize_search(self, data: Dict[str, Any]) -> SearchResult:
        """Deserializar búsqueda desde JSON"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return SearchResult(**data)


# Instancia singleton del gestor de estado
app_state_manager = AppStateManager()