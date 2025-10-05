"""
Módulo de visualización para análisis estadístico de imágenes balísticas.
Genera gráficos y visualizaciones para análisis de correlación, PCA y clustering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para usar backend no interactivo
plt.switch_backend('Agg')

# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

class StatisticalVisualizer:
    """
    Clase para generar visualizaciones de análisis estadístico
    """
    
    def __init__(self, output_dir: str = "statistical_visualizations"):
        """
        Inicializa el visualizador estadístico
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def visualize_correlation_matrix(self, correlation_data: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        Genera un mapa de calor de la matriz de correlación
        
        Args:
            correlation_data: Datos de correlación del StatisticalAnalyzer
            save_path: Ruta donde guardar la imagen
            
        Returns:
            Ruta del archivo generado
        """
        try:
            if 'correlation_matrix' not in correlation_data:
                raise ValueError("No se encontró matriz de correlación en los datos")
            
            # Convertir a DataFrame si es necesario
            corr_matrix = correlation_data['correlation_matrix']
            if isinstance(corr_matrix, dict):
                df_corr = pd.DataFrame(corr_matrix)
            else:
                df_corr = corr_matrix
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Generar mapa de calor
            mask = np.triu(np.ones_like(df_corr, dtype=bool))  # Máscara triangular superior
            heatmap = sns.heatmap(
                df_corr, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                ax=ax
            )
            
            plt.title('Matriz de Correlación de Características Balísticas', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Características', fontsize=12)
            plt.ylabel('Características', fontsize=12)
            
            # Rotar etiquetas para mejor legibilidad
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Guardar imagen
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"correlation_matrix_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Mapa de correlación guardado en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error generando mapa de correlación: {str(e)}")
            plt.close()
            raise
    
    def visualize_pca_analysis(self, pca_data: Dict[str, Any], 
                              features_data: List[Dict[str, float]] = None,
                              save_path: Optional[str] = None) -> str:
        """
        Genera visualizaciones del análisis PCA
        
        Args:
            pca_data: Datos del análisis PCA
            features_data: Datos originales de características (opcional)
            save_path: Ruta donde guardar la imagen
            
        Returns:
            Ruta del archivo generado
        """
        try:
            # Crear figura con subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Gráfico de varianza explicada
            if 'explained_variance_ratio' in pca_data:
                variance_ratio = pca_data['explained_variance_ratio']
                cumulative_variance = np.cumsum(variance_ratio)
                
                ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, 
                       alpha=0.7, color='skyblue', label='Varianza individual')
                ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                        'ro-', color='red', label='Varianza acumulada')
                ax1.set_xlabel('Componente Principal')
                ax1.set_ylabel('Proporción de Varianza Explicada')
                ax1.set_title('Varianza Explicada por Componente Principal')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Scatter plot de las dos primeras componentes
            if 'transformed_data' in pca_data:
                transformed = np.array(pca_data['transformed_data'])
                if transformed.shape[1] >= 2:
                    scatter = ax2.scatter(transformed[:, 0], transformed[:, 1], 
                                        alpha=0.6, s=50, c=range(len(transformed)), 
                                        cmap='viridis')
                    ax2.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} varianza)')
                    ax2.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} varianza)')
                    ax2.set_title('Proyección en las Primeras Dos Componentes Principales')
                    ax2.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax2, label='Índice de muestra')
            
            # 3. Importancia de características (loadings)
            if 'feature_importance' in pca_data:
                importance = pca_data['feature_importance']
                features = list(importance.keys())
                values = list(importance.values())
                
                # Tomar las 10 características más importantes
                sorted_items = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)[:10]
                features_top, values_top = zip(*sorted_items)
                
                colors = ['red' if v < 0 else 'blue' for v in values_top]
                bars = ax3.barh(range(len(features_top)), values_top, color=colors, alpha=0.7)
                ax3.set_yticks(range(len(features_top)))
                ax3.set_yticklabels(features_top)
                ax3.set_xlabel('Importancia de la Característica')
                ax3.set_title('Top 10 Características más Importantes (PC1)')
                ax3.grid(True, alpha=0.3)
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 4. Biplot (si hay datos originales)
            if features_data and 'components' in pca_data:
                # Recrear PCA para obtener los loadings
                df = pd.DataFrame(features_data)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df.fillna(0))
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Scatter de puntos
                ax4.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
                
                # Vectores de características
                feature_names = df.columns[:10]  # Limitar a 10 para claridad
                loadings = pca.components_.T[:len(feature_names), :2]
                
                for i, (feature, loading) in enumerate(zip(feature_names, loadings)):
                    ax4.arrow(0, 0, loading[0]*3, loading[1]*3, 
                             head_width=0.1, head_length=0.1, 
                             fc='red', ec='red', alpha=0.7)
                    ax4.text(loading[0]*3.2, loading[1]*3.2, feature, 
                            fontsize=8, ha='center', va='center')
                
                ax4.set_xlabel('PC1')
                ax4.set_ylabel('PC2')
                ax4.set_title('Biplot PCA')
                ax4.grid(True, alpha=0.3)
                ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            plt.suptitle('Análisis de Componentes Principales (PCA)', 
                        fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Guardar imagen
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"pca_analysis_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Análisis PCA guardado en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error generando visualización PCA: {str(e)}")
            plt.close()
            raise
    
    def visualize_clustering_analysis(self, clustering_data: Dict[str, Any],
                                    features_data: List[Dict[str, float]] = None,
                                    save_path: Optional[str] = None) -> str:
        """
        Genera visualizaciones del análisis de clustering
        
        Args:
            clustering_data: Datos del análisis de clustering
            features_data: Datos originales de características
            save_path: Ruta donde guardar la imagen
            
        Returns:
            Ruta del archivo generado
        """
        try:
            # Crear figura con subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Scatter plot de clusters (usando PCA para reducir dimensionalidad)
            if features_data and 'cluster_labels' in clustering_data:
                df = pd.DataFrame(features_data)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df.fillna(0))
                
                # Reducir dimensionalidad con PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                labels = clustering_data['cluster_labels']
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                
                for label, color in zip(unique_labels, colors):
                    if label == -1:  # Outliers en DBSCAN
                        ax1.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                                  c='black', marker='x', s=50, alpha=0.6, label='Outliers')
                    else:
                        ax1.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                                  c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
                
                ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
                ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
                ax1.set_title('Clusters en Espacio PCA')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # 2. Distribución de tamaños de clusters
            if 'cluster_sizes' in clustering_data:
                cluster_sizes = clustering_data['cluster_sizes']
                clusters = list(cluster_sizes.keys())
                sizes = list(cluster_sizes.values())
                
                bars = ax2.bar(clusters, sizes, alpha=0.7, color='lightcoral')
                ax2.set_xlabel('Cluster')
                ax2.set_ylabel('Número de Muestras')
                ax2.set_title('Distribución de Tamaños de Clusters')
                ax2.grid(True, alpha=0.3)
                
                # Añadir valores en las barras
                for bar, size in zip(bars, sizes):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(size), ha='center', va='bottom')
            
            # 3. Métricas de clustering
            metrics_text = []
            if 'silhouette_score' in clustering_data:
                metrics_text.append(f"Silhouette Score: {clustering_data['silhouette_score']:.3f}")
            if 'inertia' in clustering_data:
                metrics_text.append(f"Inertia: {clustering_data['inertia']:.2f}")
            if 'n_clusters' in clustering_data:
                metrics_text.append(f"Número de Clusters: {clustering_data['n_clusters']}")
            if 'algorithm' in clustering_data:
                metrics_text.append(f"Algoritmo: {clustering_data['algorithm']}")
            
            ax3.text(0.1, 0.9, '\n'.join(metrics_text), transform=ax3.transAxes, 
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax3.set_title('Métricas de Clustering')
            ax3.axis('off')
            
            # 4. t-SNE visualization (si hay suficientes datos)
            if features_data and len(features_data) > 10:
                try:
                    df = pd.DataFrame(features_data)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df.fillna(0))
                    
                    # Aplicar t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_data)-1))
                    X_tsne = tsne.fit_transform(X_scaled)
                    
                    labels = clustering_data['cluster_labels']
                    unique_labels = np.unique(labels)
                    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
                    
                    for label, color in zip(unique_labels, colors):
                        if label == -1:
                            ax4.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1], 
                                      c='black', marker='x', s=50, alpha=0.6, label='Outliers')
                        else:
                            ax4.scatter(X_tsne[labels == label, 0], X_tsne[labels == label, 1], 
                                      c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
                    
                    ax4.set_xlabel('t-SNE 1')
                    ax4.set_ylabel('t-SNE 2')
                    ax4.set_title('Clusters en Espacio t-SNE')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error en t-SNE:\n{str(e)}', 
                            transform=ax4.transAxes, ha='center', va='center')
                    ax4.set_title('t-SNE (Error)')
                    ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, 'Datos insuficientes\npara t-SNE', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('t-SNE (No disponible)')
                ax4.axis('off')
            
            plt.suptitle('Análisis de Clustering', fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Guardar imagen
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"clustering_analysis_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Análisis de clustering guardado en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error generando visualización de clustering: {str(e)}")
            plt.close()
            raise
    
    def generate_comprehensive_report(self, 
                                    correlation_data: Optional[Dict[str, Any]] = None,
                                    pca_data: Optional[Dict[str, Any]] = None,
                                    clustering_data: Optional[Dict[str, Any]] = None,
                                    features_data: Optional[List[Dict[str, float]]] = None,
                                    output_prefix: str = "statistical_report") -> Dict[str, str]:
        """
        Genera un reporte completo con todas las visualizaciones
        
        Args:
            correlation_data: Datos de análisis de correlación
            pca_data: Datos de análisis PCA
            clustering_data: Datos de análisis de clustering
            features_data: Datos originales de características
            output_prefix: Prefijo para los archivos de salida
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        generated_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Generar visualización de correlación
            if correlation_data:
                corr_path = os.path.join(self.output_dir, f"{output_prefix}_correlation_{timestamp}.png")
                generated_files['correlation'] = self.visualize_correlation_matrix(
                    correlation_data, corr_path)
            
            # Generar visualización PCA
            if pca_data:
                pca_path = os.path.join(self.output_dir, f"{output_prefix}_pca_{timestamp}.png")
                generated_files['pca'] = self.visualize_pca_analysis(
                    pca_data, features_data, pca_path)
            
            # Generar visualización de clustering
            if clustering_data:
                cluster_path = os.path.join(self.output_dir, f"{output_prefix}_clustering_{timestamp}.png")
                generated_files['clustering'] = self.visualize_clustering_analysis(
                    clustering_data, features_data, cluster_path)
            
            self.logger.info(f"Reporte completo generado. Archivos: {list(generated_files.keys())}")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error generando reporte completo: {str(e)}")
            return generated_files


def test_statistical_visualizer():
    """
    Función de prueba para el visualizador estadístico
    """
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    # Generar características sintéticas
    features_data = []
    for i in range(n_samples):
        sample = {
            f'feature_{j}': np.random.normal(0, 1) + (i % 3) * 2  # 3 grupos
            for j in range(n_features)
        }
        features_data.append(sample)
    
    # Crear datos de correlación sintéticos
    df = pd.DataFrame(features_data)
    correlation_data = {
        'correlation_matrix': df.corr().to_dict(),
        'high_correlations': [
            {'feature1': 'feature_0', 'feature2': 'feature_1', 'correlation': 0.85}
        ]
    }
    
    # Crear datos PCA sintéticos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    pca_data = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'transformed_data': X_pca.tolist(),
        'feature_importance': {f'feature_{i}': float(pca.components_[0][i]) 
                              for i in range(n_features)},
        'components': pca.components_.tolist()
    }
    
    # Crear datos de clustering sintéticos
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    clustering_data = {
        'cluster_labels': labels.tolist(),
        'cluster_sizes': {str(i): int(np.sum(labels == i)) for i in range(3)},
        'silhouette_score': 0.65,
        'inertia': float(kmeans.inertia_),
        'n_clusters': 3,
        'algorithm': 'KMeans'
    }
    
    # Crear visualizador y generar reporte
    visualizer = StatisticalVisualizer("test_statistical_output")
    
    print("Generando visualizaciones de prueba...")
    generated_files = visualizer.generate_comprehensive_report(
        correlation_data=correlation_data,
        pca_data=pca_data,
        clustering_data=clustering_data,
        features_data=features_data,
        output_prefix="test"
    )
    
    print("Archivos generados:")
    for analysis_type, file_path in generated_files.items():
        print(f"  {analysis_type}: {file_path}")


if __name__ == "__main__":
    test_statistical_visualizer()