#!/usr/bin/env python3
"""
Analizador Estadístico Avanzado para Análisis Balístico
Migrado al núcleo estadístico unificado con fallbacks de compatibilidad
"""

import cv2
import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_ind, chi2_contingency, kstest, shapiro,
    pearsonr, spearmanr, kendalltau
)
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Importaciones opcionales con fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Mock básico para pandas
    class MockDataFrame:
        def __init__(self, data):
            self.data = data
        def values(self):
            return np.array(self.data)
    pd = type('MockPandas', (), {'DataFrame': MockDataFrame})()

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Mocks básicos para sklearn
    class MockPCA:
        def __init__(self, n_components=None):
            self.n_components = n_components
        def fit_transform(self, X):
            return X[:, :self.n_components] if self.n_components else X
        def explained_variance_ratio_(self):
            return np.array([0.5, 0.3, 0.2])
    
    class MockStandardScaler:
        def fit_transform(self, X):
            return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    PCA = MockPCA
    StandardScaler = MockStandardScaler

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Intentar usar el núcleo estadístico unificado directamente
try:
    from common.statistical_core import UnifiedStatisticalAnalysis
    logger.info("Usando núcleo estadístico unificado directamente")
    
    # Implementación principal usando UnifiedStatisticalAnalysis
    class StatisticalAnalyzer:
        """
        Analizador estadístico migrado al núcleo unificado
        """
        def __init__(self):
            self.unified_stats = UnifiedStatisticalAnalysis()
            self.scaler = StandardScaler()
            self.pca_model = None
            self.logger = logger
        
        def perform_pca_analysis(self, features_data: List[Dict[str, float]], 
                               n_components: Optional[int] = None,
                               variance_threshold: float = 0.95) -> Dict[str, Any]:
            """
            Realiza análisis de componentes principales usando el núcleo unificado
            """
            try:
                # Convertir a DataFrame
                df = pd.DataFrame(features_data)
                
                if df.empty:
                    return {"error": "No hay datos para analizar"}
                
                # Eliminar columnas con valores constantes
                df = df.loc[:, df.var() != 0]
                
                if df.shape[1] == 0:
                    return {"error": "Todas las características son constantes"}
                
                # Normalizar datos
                X_scaled = self.scaler.fit_transform(df.fillna(0))
                
                # Determinar número de componentes automáticamente si no se especifica
                if n_components is None:
                    pca_temp = PCA()
                    pca_temp.fit(X_scaled)
                    cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
                    n_components = max(2, min(n_components, X_scaled.shape[1]))
                
                # Realizar PCA
                self.pca_model = PCA(n_components=n_components)
                X_pca = self.pca_model.fit_transform(X_scaled)
                
                # Calcular métricas adicionales
                feature_importance = self._calculate_feature_importance()
                
                result = {
                    "n_components": n_components,
                    "explained_variance": self.pca_model.explained_variance_.tolist(),
                    "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
                    "cumulative_variance_ratio": np.cumsum(self.pca_model.explained_variance_ratio_).tolist(),
                    "components": self.pca_model.components_.tolist(),
                    "transformed_data": X_pca.tolist(),
                    "feature_names": df.columns.tolist(),
                    "feature_importance": feature_importance,
                    "total_variance_explained": float(np.sum(self.pca_model.explained_variance_ratio_)),
                    "n_samples": X_scaled.shape[0],
                    "n_features_original": X_scaled.shape[1]
                }
                
                self.logger.info(f"PCA completado: {n_components} componentes, "
                               f"{result['total_variance_explained']:.3f} varianza explicada")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error en PCA: {str(e)}")
                return {"error": str(e)}
        
        def perform_significance_tests(self, group1: List[float], group2: List[float],
                                     alpha: float = 0.05) -> Dict[str, Any]:
            """
            Realiza pruebas de significancia usando el núcleo unificado
            """
            try:
                # Usar el núcleo unificado para las pruebas
                t_test_result = self.unified_stats.calculate_p_value(group1, group2, 't_test')
                mw_result = self.unified_stats.calculate_p_value(group1, group2, 'mann_whitney')
                ks_result = self.unified_stats.calculate_p_value(group1, group2, 'kolmogorov_smirnov')
                
                return {
                     't_test': {
                         'statistic': float(t_test_result.statistic),
                         'p_value': float(t_test_result.p_value),
                         'is_significant': bool(t_test_result.p_value < alpha)
                     },
                     'mann_whitney': {
                         'statistic': float(mw_result.statistic),
                         'p_value': float(mw_result.p_value),
                         'is_significant': bool(mw_result.p_value < alpha)
                     },
                     'kolmogorov_smirnov': {
                         'statistic': float(ks_result.statistic),
                         'p_value': float(ks_result.p_value),
                         'is_significant': bool(ks_result.p_value < alpha)
                     }
                 }
                
            except Exception as e:
                self.logger.error(f"Error en pruebas de significancia: {str(e)}")
                return {"error": str(e)}
        
        def _calculate_feature_importance(self) -> Dict[str, float]:
            """
            Calcula la importancia de las características basada en los componentes PCA
            """
            if self.pca_model is None:
                return {}
            
            try:
                # Calcular importancia como suma de valores absolutos de componentes
                components = np.abs(self.pca_model.components_)
                feature_importance = np.sum(components, axis=0)
                
                # Normalizar
                if np.sum(feature_importance) > 0:
                    feature_importance = feature_importance / np.sum(feature_importance)
                
                return {f"feature_{i}": float(imp) for i, imp in enumerate(feature_importance)}
                
            except Exception as e:
                self.logger.error(f"Error calculando importancia: {str(e)}")
                return {}
        
        def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
            """
            Analiza una imagen y extrae características estadísticas
            """
            try:
                if image is None or image.size == 0:
                    return {"error": "Imagen inválida"}
                
                # Convertir a escala de grises si es necesario
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Calcular estadísticas básicas
                mean_val = float(np.mean(gray))
                std_val = float(np.std(gray))
                entropy = self._calculate_entropy(gray)
                contrast = self._calculate_contrast(gray)
                
                return {
                    "mean": mean_val,
                    "std": std_val,
                    "entropy": entropy,
                    "contrast": contrast,
                    "shape": gray.shape
                }
                
            except Exception as e:
                self.logger.error(f"Error analizando imagen: {str(e)}")
                return {"error": str(e)}
        
        def _calculate_entropy(self, image: np.ndarray) -> float:
            """
            Calcula la entropía de una imagen
            """
            try:
                hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
                hist = hist[hist > 0]  # Eliminar bins vacíos
                prob = hist / np.sum(hist)
                entropy = -np.sum(prob * np.log2(prob))
                return float(entropy)
            except:
                return 0.0
        
        def _calculate_contrast(self, image: np.ndarray) -> float:
            """
            Calcula el contraste de una imagen usando la desviación estándar
            """
            try:
                return float(np.std(image))
            except:
                return 0.0

except ImportError as e:
    logger.warning(f"Fallback a adaptador de compatibilidad: {e}")
    
    # Fallback al adaptador de compatibilidad
    # Importación diferida para evitar dependencia circular
    try:
        # Importar solo cuando sea necesario
        def _get_statistical_adapter():
            from common.compatibility_adapters import StatisticalAnalyzerAdapter
            return StatisticalAnalyzerAdapter
        
        logger.info("Usando StatisticalAnalyzerAdapter del núcleo unificado (importación diferida)")
        
        # Clase que usa importación diferida
        class StatisticalAnalyzer:
            """
            Analizador estadístico con importación diferida para evitar dependencias circulares
            """
            
            def __init__(self):
                self._adapter = None
            
            def _get_adapter(self):
                if self._adapter is None:
                    StatisticalAnalyzerAdapter = _get_statistical_adapter()
                    self._adapter = StatisticalAnalyzerAdapter()
                return self._adapter
            
            def perform_pca_analysis(self, features_data: List[Dict[str, float]], 
                                   n_components: Optional[int] = None,
                                   variance_threshold: float = 0.95) -> Dict[str, Any]:
                return self._get_adapter().perform_pca_analysis(features_data, n_components, variance_threshold)
            
            def perform_significance_tests(self, group1: List[float], group2: List[float],
                                         alpha: float = 0.05) -> Dict[str, Any]:
                return self._get_adapter().perform_significance_tests(group1, group2, alpha)
            
            def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
                return self._get_adapter().analyze_image(image)
            
            def _calculate_entropy(self, image: np.ndarray) -> float:
                return self._get_adapter()._calculate_entropy(image)
            
            def _calculate_contrast(self, image: np.ndarray) -> float:
                return self._get_adapter()._calculate_contrast(image)
            
    except ImportError as e:
        logger.warning(f"Fallback a implementación original: {e}")
        
        # Implementación original como fallback
        class StatisticalAnalyzer:
            """
            Implementación fallback básica de StatisticalAnalyzer
            """
            def __init__(self):
                self.scaler = StandardScaler()
                self.pca_model = None
                self.logger = logger
        
            def perform_pca_analysis(self, features_data: List[Dict[str, float]], 
                                   n_components: Optional[int] = None,
                                   variance_threshold: float = 0.95) -> Dict[str, Any]:
                """
                Realiza análisis de componentes principales
                """
                try:
                    # Convertir a DataFrame
                    df = pd.DataFrame(features_data)
                    
                    if df.empty:
                        return {"error": "No hay datos para analizar"}
                    
                    # Eliminar columnas con valores constantes
                    df = df.loc[:, df.var() != 0]
                    
                    if df.shape[1] == 0:
                        return {"error": "Todas las características son constantes"}
                    
                    # Normalizar datos
                    X_scaled = self.scaler.fit_transform(df.fillna(0))
                    
                    # Determinar número de componentes automáticamente si no se especifica
                    if n_components is None:
                        pca_temp = PCA()
                        pca_temp.fit(X_scaled)
                        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
                        n_components = max(2, min(n_components, X_scaled.shape[1]))
                    
                    # Realizar PCA
                    self.pca_model = PCA(n_components=n_components)
                    X_pca = self.pca_model.fit_transform(X_scaled)
                    
                    # Calcular métricas adicionales
                    feature_importance = self._calculate_feature_importance()
                    
                    result = {
                        "n_components": n_components,
                        "explained_variance": self.pca_model.explained_variance_.tolist(),
                        "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
                        "cumulative_variance_ratio": np.cumsum(self.pca_model.explained_variance_ratio_).tolist(),
                        "components": self.pca_model.components_.tolist(),
                        "transformed_data": X_pca.tolist(),
                        "feature_names": df.columns.tolist(),
                        "feature_importance": feature_importance,
                        "total_variance_explained": float(np.sum(self.pca_model.explained_variance_ratio_)),
                        "n_samples": X_scaled.shape[0],
                        "n_features_original": X_scaled.shape[1]
                    }
                    
                    self.logger.info(f"PCA completado: {n_components} componentes, "
                                   f"{result['total_variance_explained']:.3f} varianza explicada")
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Error en PCA: {str(e)}")
                    return {"error": str(e)}
            
            def perform_significance_tests(self, group1: List[float], group2: List[float],
                                         alpha: float = 0.05) -> Dict[str, Any]:
                """
                Realiza pruebas de significancia estadística entre dos grupos
                """
                try:
                    # Convertir a arrays numpy
                    data1 = np.array(group1, dtype=float)
                    data2 = np.array(group2, dtype=float)
                    
                    # Validar datos
                    if len(data1) == 0 or len(data2) == 0:
                        return {"error": "Los grupos no pueden estar vacíos"}
                    
                    results = {}
                    
                    # Test t de Student
                    try:
                        t_stat, t_p = stats.ttest_ind(data1, data2)
                        results['t_test'] = {
                             'statistic': float(t_stat),
                             'p_value': float(t_p),
                             'is_significant': bool(t_p < alpha)
                         }
                    except Exception as e:
                        results['t_test'] = {'error': str(e)}
                    
                    # Test Mann-Whitney U
                    try:
                        mw_stat, mw_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        results['mann_whitney'] = {
                             'statistic': float(mw_stat),
                             'p_value': float(mw_p),
                             'is_significant': bool(mw_p < alpha)
                         }
                    except Exception as e:
                        results['mann_whitney'] = {'error': str(e)}
                    
                    # Test Kolmogorov-Smirnov
                    try:
                        ks_stat, ks_p = stats.ks_2samp(data1, data2)
                        results['kolmogorov_smirnov'] = {
                             'statistic': float(ks_stat),
                             'p_value': float(ks_p),
                             'is_significant': bool(ks_p < alpha)
                         }
                    except Exception as e:
                        results['kolmogorov_smirnov'] = {'error': str(e)}
                    
                    return results
                    
                except Exception as e:
                    self.logger.error(f"Error en pruebas de significancia: {str(e)}")
                    return {"error": str(e)}
            
            def _calculate_feature_importance(self) -> Dict[str, float]:
                """
                Calcula la importancia de las características basada en los componentes PCA
                """
                if self.pca_model is None:
                    return {}
                
                try:
                    # Calcular importancia como suma de valores absolutos de componentes
                    components = np.abs(self.pca_model.components_)
                    feature_importance = np.sum(components, axis=0)
                    
                    # Normalizar
                    if np.sum(feature_importance) > 0:
                        feature_importance = feature_importance / np.sum(feature_importance)
                    
                    return {f"feature_{i}": float(imp) for i, imp in enumerate(feature_importance)}
                    
                except Exception as e:
                    self.logger.error(f"Error calculando importancia: {str(e)}")
                    return {}
            
            def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
                """
                Analiza una imagen y extrae características estadísticas
                """
                try:
                    if image is None or image.size == 0:
                        return {"error": "Imagen inválida"}
                    
                    # Convertir a escala de grises si es necesario
                    if len(image.shape) == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                    
                    # Calcular estadísticas básicas
                    mean_val = float(np.mean(gray))
                    std_val = float(np.std(gray))
                    entropy = self._calculate_entropy(gray)
                    contrast = self._calculate_contrast(gray)
                    
                    return {
                        "mean": mean_val,
                        "std": std_val,
                        "entropy": entropy,
                        "contrast": contrast,
                        "shape": gray.shape
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error analizando imagen: {str(e)}")
                    return {"error": str(e)}
            
            def _calculate_entropy(self, image: np.ndarray) -> float:
                """
                Calcula la entropía de una imagen
                """
                try:
                    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
                    hist = hist[hist > 0]  # Eliminar bins vacíos
                    prob = hist / np.sum(hist)
                    entropy = -np.sum(prob * np.log2(prob))
                    return float(entropy)
                except:
                    return 0.0
            
            def _calculate_contrast(self, image: np.ndarray) -> float:
                """
                Calcula el contraste de una imagen usando la desviación estándar
                """
                try:
                    return float(np.std(image))
                except:
                    return 0.0


def main():
    """
    Función principal para pruebas del analizador estadístico
    """
    print("=== Prueba de Análisis PCA ===")
    
    # Crear datos de prueba
    test_data = [
        {"feature_1": 1.2, "feature_2": 2.3, "feature_3": 3.1},
        {"feature_1": 1.5, "feature_2": 2.1, "feature_3": 3.4},
        {"feature_1": 1.1, "feature_2": 2.5, "feature_3": 2.9},
        {"feature_1": 1.8, "feature_2": 1.9, "feature_3": 3.2},
        {"feature_1": 1.3, "feature_2": 2.4, "feature_3": 3.0}
    ]
    
    analyzer = StatisticalAnalyzer()
    pca_result = analyzer.perform_pca_analysis(test_data, n_components=2)
    print(f"Resultado PCA: {json.dumps(pca_result, indent=2)}")
    
    print("\n=== Prueba de Tests de Significancia ===")
    
    # Datos de prueba para tests estadísticos
    group1 = [1.2, 1.5, 1.1, 1.8, 1.3, 1.4, 1.6]
    group2 = [2.1, 2.3, 1.9, 2.5, 2.2, 2.0, 2.4]
    
    sig_result = analyzer.perform_significance_tests(group1, group2)
    print(f"Resultado Tests: {json.dumps(sig_result, indent=2)}")


if __name__ == "__main__":
    main()