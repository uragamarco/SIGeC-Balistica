#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common Integration Module - SIGeC-Balistica GUI
===========================================

Módulo de integración específico para componentes comunes del sistema,
proporcionando acceso al núcleo estadístico unificado desde la GUI.

Funcionalidades:
- Análisis estadístico unificado
- Integración NIST
- Adaptadores de compatibilidad
- Bootstrap y análisis estadístico avanzado

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

# Importaciones de componentes common
try:
    from common.statistical_core import (
        UnifiedStatisticalAnalysis,
        StatisticalCore,
        BootstrapResult,
        StatisticalTestResult,
        MultipleComparisonResult,
        SimilarityBootstrapResult,
        MatchingBootstrapConfig,
        StatisticalTest,
        CorrectionMethod,
        create_bootstrap_adapter,
        create_statistical_adapter,
        create_similarity_bootstrap_function,
        calculate_bootstrap_confidence_interval
    )
    from common.compatibility_adapters import (
        AdvancedStatisticalAnalysisAdapter,
        BootstrapSimilarityAnalyzerAdapter,
        StatisticalAnalyzerAdapter
    )
    from common.nist_integration import NISTStatisticalIntegration
    COMMON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Módulo common no disponible: {e}")
    COMMON_AVAILABLE = False

class CommonIntegration(QObject):
    """
    Clase de integración para componentes comunes del sistema
    """
    
    # Señales para comunicación con la GUI
    analysis_started = pyqtSignal(str)  # analysis_id
    analysis_completed = pyqtSignal(str, dict)  # analysis_id, results
    analysis_error = pyqtSignal(str, str)  # analysis_id, error_message
    bootstrap_progress = pyqtSignal(str, int)  # analysis_id, progress_percentage
    
    def __init__(self):
        super().__init__()
        
        self.unified_statistical_analysis = None
        self.statistical_core = None
        self.nist_integration = None
        
        # Adaptadores de compatibilidad
        self.advanced_stats_adapter = None
        self.bootstrap_similarity_adapter = None
        self.statistical_analyzer_adapter = None
        
        self.active_analyses = {}
        
        if COMMON_AVAILABLE:
            self._initialize_common_components()
    
    def _initialize_common_components(self):
        """Inicializa los componentes common"""
        try:
            # Núcleo estadístico unificado
            self.unified_statistical_analysis = UnifiedStatisticalAnalysis()
            self.statistical_core = StatisticalCore()
            
            # Integración NIST
            self.nist_integration = NISTStatisticalIntegration()
            
            # Adaptadores de compatibilidad
            self.advanced_stats_adapter = AdvancedStatisticalAnalysisAdapter()
            self.bootstrap_similarity_adapter = BootstrapSimilarityAnalyzerAdapter()
            self.statistical_analyzer_adapter = StatisticalAnalyzerAdapter()
            
            logging.info("Componentes common inicializados correctamente")
            
        except Exception as e:
            logging.error(f"Error inicializando componentes common: {e}")
    
    def perform_statistical_analysis(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Realiza análisis estadístico unificado
        
        Args:
            data: Datos para análisis
            config: Configuración del análisis
            
        Returns:
            ID del análisis o None si hay error
        """
        if not COMMON_AVAILABLE or not self.unified_statistical_analysis:
            return self._fallback_statistical_analysis(data, config)
        
        try:
            import uuid
            analysis_id = str(uuid.uuid4())
            
            self.analysis_started.emit(analysis_id)
            self.active_analyses[analysis_id] = {'data': data, 'config': config or {}}
            
            # Realizar análisis estadístico
            results = self.unified_statistical_analysis.analyze(data, config or {})
            
            self.analysis_completed.emit(analysis_id, results)
            return analysis_id
            
        except Exception as e:
            error_msg = f"Error en análisis estadístico: {e}"
            self.analysis_error.emit(analysis_id if 'analysis_id' in locals() else 'unknown', error_msg)
            return None
    
    def _fallback_statistical_analysis(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Análisis estadístico básico sin componentes common"""
        import uuid
        analysis_id = str(uuid.uuid4())
        
        self.analysis_started.emit(analysis_id)
        
        # Simular análisis básico
        results = {
            'status': 'completed',
            'message': 'Análisis estadístico básico completado',
            'data_summary': {
                'sample_count': len(data.get('samples', [])),
                'features': list(data.keys())
            },
            'basic_stats': self._calculate_basic_stats(data)
        }
        
        self.analysis_completed.emit(analysis_id, results)
        return analysis_id
    
    def _calculate_basic_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula estadísticas básicas sin componentes common"""
        try:
            import numpy as np
            
            stats = {}
            for key, values in data.items():
                if isinstance(values, (list, tuple)) and len(values) > 0:
                    if all(isinstance(v, (int, float)) for v in values):
                        arr = np.array(values)
                        stats[key] = {
                            'mean': float(np.mean(arr)),
                            'std': float(np.std(arr)),
                            'min': float(np.min(arr)),
                            'max': float(np.max(arr)),
                            'count': len(arr)
                        }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculando estadísticas básicas: {e}")
            return {}
    
    def perform_bootstrap_analysis(self, data: List[float], config: Optional[MatchingBootstrapConfig] = None) -> Optional[BootstrapResult]:
        """
        Realiza análisis bootstrap
        
        Args:
            data: Datos para bootstrap
            config: Configuración del bootstrap
            
        Returns:
            Resultado del bootstrap o None si hay error
        """
        if not COMMON_AVAILABLE or not self.unified_statistical_analysis:
            return self._fallback_bootstrap_analysis(data, config)
        
        try:
            # Usar configuración por defecto si no se proporciona
            if config is None:
                config = MatchingBootstrapConfig()
            
            # Usar bootstrap_sampling del UnifiedStatisticalAnalysis
            import numpy as np
            result = self.unified_statistical_analysis.bootstrap_sampling(
                data=data,
                statistic_func=np.mean,
                n_bootstrap=config.n_bootstrap if hasattr(config, 'n_bootstrap') else 1000,
                confidence_level=config.confidence_level if hasattr(config, 'confidence_level') else 0.95
            )
            return result
            
        except Exception as e:
            logging.error(f"Error en análisis bootstrap: {e}")
            return None
    
    def _fallback_bootstrap_analysis(self, data: List[float], config: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Análisis bootstrap básico sin componentes common"""
        try:
            import numpy as np
            
            # Bootstrap básico
            n_bootstrap = 1000 if config is None else getattr(config, 'n_bootstrap', 1000)
            bootstrap_samples = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_samples.append(np.mean(sample))
            
            bootstrap_samples = np.array(bootstrap_samples)
            
            return {
                'mean': float(np.mean(bootstrap_samples)),
                'std': float(np.std(bootstrap_samples)),
                'confidence_interval': {
                    'lower': float(np.percentile(bootstrap_samples, 2.5)),
                    'upper': float(np.percentile(bootstrap_samples, 97.5))
                },
                'samples': bootstrap_samples.tolist()
            }
            
        except Exception as e:
            logging.error(f"Error en bootstrap básico: {e}")
            return None
    
    def perform_statistical_test(self, data1: List[float], data2: List[float], test_type: str = 'ttest') -> Optional[StatisticalTestResult]:
        """
        Realiza test estadístico entre dos grupos
        
        Args:
            data1: Primer grupo de datos
            data2: Segundo grupo de datos
            test_type: Tipo de test ('ttest', 'mannwhitney', 'ks')
            
        Returns:
            Resultado del test o None si hay error
        """
        if not COMMON_AVAILABLE or not self.statistical_core:
            return self._fallback_statistical_test(data1, data2, test_type)
        
        try:
            # Mapear tipo de test
            test_map = {
                'ttest': StatisticalTest.TTEST,
                'mannwhitney': StatisticalTest.MANN_WHITNEY,
                'ks': StatisticalTest.KOLMOGOROV_SMIRNOV
            }
            
            test_enum = test_map.get(test_type, StatisticalTest.TTEST)
            result = self.statistical_core.statistical_test(data1, data2, test_enum)
            return result
            
        except Exception as e:
            logging.error(f"Error en test estadístico: {e}")
            return None
    
    def _fallback_statistical_test(self, data1: List[float], data2: List[float], test_type: str) -> Optional[Dict[str, Any]]:
        """Test estadístico básico sin componentes common"""
        try:
            from scipy import stats
            
            if test_type == 'ttest':
                statistic, p_value = stats.ttest_ind(data1, data2)
            elif test_type == 'mannwhitney':
                statistic, p_value = stats.mannwhitneyu(data1, data2)
            elif test_type == 'ks':
                statistic, p_value = stats.ks_2samp(data1, data2)
            else:
                statistic, p_value = stats.ttest_ind(data1, data2)
            
            return {
                'test_type': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            logging.error(f"Error en test estadístico básico: {e}")
            return None
    
    def get_nist_compliance_report(self, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Genera reporte de cumplimiento NIST
        
        Args:
            analysis_results: Resultados del análisis
            
        Returns:
            Reporte de cumplimiento NIST o None si hay error
        """
        if not COMMON_AVAILABLE or not self.nist_integration:
            return self._fallback_nist_compliance(analysis_results)
        
        try:
            # Usar analyze_nist_compliance en lugar de generate_compliance_report
            # Crear un mock quality_report si no está disponible
            mock_quality_report = {
                'overall_quality': analysis_results.get('quality_score', 0.8),
                'metrics': analysis_results
            }
            
            nist_report = self.nist_integration.analyze_nist_compliance(mock_quality_report)
            
            # Convertir a diccionario usando export_compliance_report
            report_dict = self.nist_integration.export_compliance_report(nist_report)
            return report_dict
            
        except Exception as e:
            logging.error(f"Error generando reporte NIST: {e}")
            return None
    
    def _fallback_nist_compliance(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Reporte NIST básico sin componentes common"""
        return {
            'compliance_status': 'basic_check',
            'nist_available': False,
            'basic_validation': {
                'has_results': bool(analysis_results),
                'result_count': len(analysis_results),
                'timestamp': str(Path(__file__).stat().st_mtime)
            },
            'recommendations': [
                'Instalar módulo common para cumplimiento NIST completo',
                'Verificar configuración de análisis estadístico'
            ]
        }
    
    def create_compatibility_adapter(self, adapter_type: str) -> Optional[Any]:
        """
        Crea adaptador de compatibilidad
        
        Args:
            adapter_type: Tipo de adaptador ('advanced_stats', 'bootstrap_similarity', 'statistical_analyzer')
            
        Returns:
            Adaptador creado o None si hay error
        """
        if not COMMON_AVAILABLE:
            return None
        
        try:
            if adapter_type == 'advanced_stats':
                return self.advanced_stats_adapter
            elif adapter_type == 'bootstrap_similarity':
                return self.bootstrap_similarity_adapter
            elif adapter_type == 'statistical_analyzer':
                return self.statistical_analyzer_adapter
            else:
                logging.warning(f"Tipo de adaptador desconocido: {adapter_type}")
                return None
                
        except Exception as e:
            logging.error(f"Error creando adaptador {adapter_type}: {e}")
            return None
    
    def get_available_statistical_tests(self) -> List[str]:
        """
        Obtiene lista de tests estadísticos disponibles
        
        Returns:
            Lista de tests disponibles
        """
        if not COMMON_AVAILABLE:
            return ['ttest', 'mannwhitney', 'ks']  # Tests básicos
        
        try:
            # Obtener tests del enum StatisticalTest
            return [test.value for test in StatisticalTest]
        except Exception as e:
            logging.error(f"Error obteniendo tests estadísticos: {e}")
            return ['ttest', 'mannwhitney', 'ks']
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Obtiene información de salud del sistema common
        
        Returns:
            Diccionario con información de salud
        """
        health_data = {
            'common_available': COMMON_AVAILABLE,
            'components': {
                'unified_statistical_analysis': self.unified_statistical_analysis is not None,
                'statistical_core': self.statistical_core is not None,
                'nist_integration': self.nist_integration is not None,
                'advanced_stats_adapter': self.advanced_stats_adapter is not None,
                'bootstrap_similarity_adapter': self.bootstrap_similarity_adapter is not None,
                'statistical_analyzer_adapter': self.statistical_analyzer_adapter is not None
            },
            'active_analyses': len(self.active_analyses),
            'available_tests': self.get_available_statistical_tests()
        }
        
        return health_data

# Instancia global
_common_integration = None

def get_common_integration() -> CommonIntegration:
    """Obtiene la instancia global de integración common"""
    global _common_integration
    if _common_integration is None:
        _common_integration = CommonIntegration()
    return _common_integration

if __name__ == "__main__":
    # Prueba básica del módulo
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    common_int = CommonIntegration()
    print("=== Common Integration Test ===")
    print("Salud del sistema:", common_int.get_system_health())
    print("Tests disponibles:", common_int.get_available_statistical_tests())
    
    # Prueba de análisis estadístico básico
    test_data = {
        'samples': [1.2, 1.5, 1.8, 2.1, 1.9, 1.7, 1.6],
        'scores': [0.85, 0.92, 0.78, 0.88, 0.91, 0.86, 0.89]
    }
    
    analysis_id = common_int.perform_statistical_analysis(test_data)
    print(f"Análisis ejecutado: {analysis_id}")
    
    # Prueba de bootstrap
    bootstrap_data = [1.2, 1.5, 1.8, 2.1, 1.9, 1.7, 1.6, 1.4, 1.3, 2.0]
    bootstrap_result = common_int.perform_bootstrap_analysis(bootstrap_data)
    print(f"Bootstrap resultado: {bootstrap_result}")