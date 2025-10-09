"""
Módulo de visualización para análisis estadístico de imágenes balísticas.
Genera gráficos y visualizaciones para análisis de correlación, PCA y clustering.
Incluye soporte para visualizaciones interactivas con Plotly.
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

# Importaciones opcionales para interactividad
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.layouts import column, row
    from bokeh.io import curdoc
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

# Configurar matplotlib para usar backend no interactivo
plt.switch_backend('Agg')

# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

class StatisticalVisualizer:
    """
    Clase para generar visualizaciones de análisis estadístico
    Soporta tanto visualizaciones estáticas como interactivas
    """
    
    def __init__(self, output_dir: str = "statistical_visualizations", 
                 interactive_mode: bool = True):
        """
        Inicializa el visualizador estadístico
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
            interactive_mode: Si usar visualizaciones interactivas cuando sea posible
        """
        self.output_dir = output_dir
        self.interactive_mode = interactive_mode and PLOTLY_AVAILABLE
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
        
        if self.interactive_mode:
            self.logger.info("Modo interactivo habilitado con Plotly")
        else:
            self.logger.info("Usando visualizaciones estáticas con Matplotlib")
    
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

    def create_interactive_correlation_heatmap(self, correlation_data: Dict[str, Any], 
                                             save_path: Optional[str] = None) -> str:
        """
        Crea un mapa de calor interactivo de correlación usando Plotly
        
        Args:
            correlation_data: Datos de correlación
            save_path: Ruta donde guardar el archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly no disponible, usando visualización estática")
            return self.visualize_correlation_matrix(correlation_data, save_path)
        
        try:
            corr_matrix = correlation_data['correlation_matrix']
            if isinstance(corr_matrix, dict):
                df_corr = pd.DataFrame(corr_matrix)
            else:
                df_corr = corr_matrix
            
            # Crear heatmap interactivo
            fig = go.Figure(data=go.Heatmap(
                z=df_corr.values,
                x=df_corr.columns,
                y=df_corr.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(df_corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlación: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': 'Matriz de Correlación Interactiva - Características Balísticas',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title="Características",
                yaxis_title="Características",
                width=800,
                height=600,
                font=dict(size=12)
            )
            
            # Guardar archivo HTML
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"interactive_correlation_{timestamp}.html")
            
            pyo.plot(fig, filename=save_path, auto_open=False)
            self.logger.info(f"Mapa de calor interactivo guardado en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creando mapa de calor interactivo: {str(e)}")
            return self.visualize_correlation_matrix(correlation_data, save_path)

    def create_interactive_clustering_plot(self, clustering_data: Dict[str, Any],
                                         features_data: List[Dict[str, float]] = None,
                                         save_path: Optional[str] = None) -> str:
        """
        Crea visualización interactiva de clustering usando Plotly
        
        Args:
            clustering_data: Datos del análisis de clustering
            features_data: Datos originales de características
            save_path: Ruta donde guardar el archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly no disponible, usando visualización estática")
            return self.visualize_clustering_analysis(clustering_data, features_data, save_path)
        
        try:
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Clusters en Espacio PCA', 'Distribución de Clusters',
                              'Métricas de Clustering', 'Clusters en Espacio t-SNE'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "table"}, {"type": "scatter"}]]
            )
            
            if features_data and 'cluster_labels' in clustering_data:
                df = pd.DataFrame(features_data)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df.fillna(0))
                
                # PCA para visualización
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                labels = clustering_data['cluster_labels']
                unique_labels = np.unique(labels)
                
                # Scatter plot PCA
                for label in unique_labels:
                    mask = labels == label
                    if label == -1:  # Outliers
                        fig.add_trace(
                            go.Scatter(
                                x=X_pca[mask, 0], y=X_pca[mask, 1],
                                mode='markers',
                                name='Outliers',
                                marker=dict(color='black', symbol='x', size=8),
                                hovertemplate='<b>Outlier</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=X_pca[mask, 0], y=X_pca[mask, 1],
                                mode='markers',
                                name=f'Cluster {label}',
                                marker=dict(size=8),
                                hovertemplate=f'<b>Cluster {label}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                
                # Actualizar ejes PCA
                fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', row=1, col=1)
                fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', row=1, col=1)
            
            # Distribución de clusters
            if 'cluster_sizes' in clustering_data:
                cluster_sizes = clustering_data['cluster_sizes']
                clusters = list(cluster_sizes.keys())
                sizes = list(cluster_sizes.values())
                
                fig.add_trace(
                    go.Bar(
                        x=clusters, y=sizes,
                        name='Tamaño de Clusters',
                        marker_color='lightcoral',
                        hovertemplate='<b>Cluster %{x}</b><br>Muestras: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text='Cluster', row=1, col=2)
                fig.update_yaxes(title_text='Número de Muestras', row=1, col=2)
            
            # Tabla de métricas
            metrics_data = []
            if 'silhouette_score' in clustering_data:
                metrics_data.append(['Silhouette Score', f"{clustering_data['silhouette_score']:.3f}"])
            if 'inertia' in clustering_data:
                metrics_data.append(['Inertia', f"{clustering_data['inertia']:.2f}"])
            if 'n_clusters' in clustering_data:
                metrics_data.append(['Número de Clusters', str(clustering_data['n_clusters'])])
            if 'algorithm' in clustering_data:
                metrics_data.append(['Algoritmo', clustering_data['algorithm']])
            
            if metrics_data:
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Métrica', 'Valor'],
                                  fill_color='lightblue',
                                  align='left'),
                        cells=dict(values=list(zip(*metrics_data)),
                                 fill_color='white',
                                 align='left')
                    ),
                    row=2, col=1
                )
            
            # t-SNE visualization
            if features_data and len(features_data) > 10:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_data)-1))
                    X_tsne = tsne.fit_transform(X_scaled)
                    
                    labels = clustering_data['cluster_labels']
                    unique_labels = np.unique(labels)
                    
                    for label in unique_labels:
                        mask = labels == label
                        if label == -1:
                            fig.add_trace(
                                go.Scatter(
                                    x=X_tsne[mask, 0], y=X_tsne[mask, 1],
                                    mode='markers',
                                    name='Outliers (t-SNE)',
                                    marker=dict(color='black', symbol='x', size=8),
                                    hovertemplate='<b>Outlier</b><br>t-SNE1: %{x:.2f}<br>t-SNE2: %{y:.2f}<extra></extra>',
                                    showlegend=False
                                ),
                                row=2, col=2
                            )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=X_tsne[mask, 0], y=X_tsne[mask, 1],
                                    mode='markers',
                                    name=f'Cluster {label} (t-SNE)',
                                    marker=dict(size=8),
                                    hovertemplate=f'<b>Cluster {label}</b><br>t-SNE1: %{{x:.2f}}<br>t-SNE2: %{{y:.2f}}<extra></extra>',
                                    showlegend=False
                                ),
                                row=2, col=2
                            )
                    
                    fig.update_xaxes(title_text='t-SNE 1', row=2, col=2)
                    fig.update_yaxes(title_text='t-SNE 2', row=2, col=2)
                    
                except Exception as e:
                    self.logger.warning(f"Error en t-SNE: {str(e)}")
            
            # Configuración general
            fig.update_layout(
                title={
                    'text': 'Análisis Interactivo de Clustering',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                height=800,
                showlegend=True
            )
            
            # Guardar archivo HTML
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"interactive_clustering_{timestamp}.html")
            
            pyo.plot(fig, filename=save_path, auto_open=False)
            self.logger.info(f"Visualización interactiva de clustering guardada en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creando visualización interactiva de clustering: {str(e)}")
            return self.visualize_clustering_analysis(clustering_data, features_data, save_path)

    def create_interactive_pca_plot(self, pca_data: Dict[str, Any],
                                  features_data: List[Dict[str, float]] = None,
                                  save_path: Optional[str] = None) -> str:
        """
        Crea visualización interactiva de PCA usando Plotly
        
        Args:
            pca_data: Datos del análisis PCA
            features_data: Datos originales de características
            save_path: Ruta donde guardar el archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly no disponible, usando visualización estática")
            return self.visualize_pca_analysis(pca_data, features_data, save_path)

    def create_interactive_dashboard(self, analysis_results: Dict[str, Any],
                                   features_data: List[Dict[str, float]] = None,
                                   save_path: Optional[str] = None) -> str:
        """
        Crea un dashboard interactivo completo con todos los análisis
        
        Args:
            analysis_results: Resultados de todos los análisis
            features_data: Datos originales de características
            save_path: Ruta donde guardar el archivo HTML
            
        Returns:
            Ruta del archivo HTML generado
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly no disponible, generando reportes estáticos")
            return self.generate_comprehensive_report(analysis_results, features_data)
        
        try:
            # Crear dashboard con múltiples pestañas
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dashboard Interactivo - Análisis Balístico</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                    .header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                    .tabs { display: flex; background-color: white; border-radius: 10px; overflow: hidden; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
                    .tab { flex: 1; padding: 15px; text-align: center; cursor: pointer; 
                          background-color: #f8f9fa; border-right: 1px solid #dee2e6; }
                    .tab:last-child { border-right: none; }
                    .tab.active { background-color: #007bff; color: white; }
                    .tab:hover { background-color: #e9ecef; }
                    .tab.active:hover { background-color: #0056b3; }
                    .content { background-color: white; padding: 20px; border-radius: 10px; 
                              box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .tab-content { display: none; }
                    .tab-content.active { display: block; }
                    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                                   gap: 20px; margin-bottom: 30px; }
                    .summary-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                   color: white; padding: 20px; border-radius: 10px; text-align: center; }
                    .summary-card h3 { margin: 0 0 10px 0; }
                    .summary-card .value { font-size: 2em; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎯 Dashboard Interactivo - Análisis Balístico SIGeC</h1>
                    <p>Análisis Estadístico Avanzado de Características Balísticas</p>
                </div>
                
                <div class="summary-grid">
            """
            
            # Agregar tarjetas de resumen
            if 'clustering' in analysis_results:
                clustering_data = analysis_results['clustering']
                dashboard_html += f"""
                    <div class="summary-card">
                        <h3>Clustering</h3>
                        <div class="value">{clustering_data.get('n_clusters', 'N/A')}</div>
                        <p>Clusters identificados</p>
                    </div>
                """
            
            if 'pca' in analysis_results:
                pca_data = analysis_results['pca']
                variance_explained = sum(pca_data.get('explained_variance_ratio', [0, 0])[:2])
                dashboard_html += f"""
                    <div class="summary-card">
                        <h3>PCA</h3>
                        <div class="value">{variance_explained:.1%}</div>
                        <p>Varianza explicada (PC1+PC2)</p>
                    </div>
                """
            
            if 'correlation' in analysis_results:
                dashboard_html += f"""
                    <div class="summary-card">
                        <h3>Correlaciones</h3>
                        <div class="value">✓</div>
                        <p>Matriz de correlación</p>
                    </div>
                """
            
            if features_data:
                dashboard_html += f"""
                    <div class="summary-card">
                        <h3>Muestras</h3>
                        <div class="value">{len(features_data)}</div>
                        <p>Total de muestras</p>
                    </div>
                """
            
            # Pestañas de navegación
            dashboard_html += """
                </div>
                
                <div class="tabs">
                    <div class="tab active" onclick="showTab('overview')">📊 Resumen</div>
            """
            
            if 'correlation' in analysis_results:
                dashboard_html += '<div class="tab" onclick="showTab(\'correlation\')">🔗 Correlaciones</div>'
            if 'clustering' in analysis_results:
                dashboard_html += '<div class="tab" onclick="showTab(\'clustering\')">🎯 Clustering</div>'
            if 'pca' in analysis_results:
                dashboard_html += '<div class="tab" onclick="showTab(\'pca\')">📈 PCA</div>'
            
            dashboard_html += """
                </div>
                
                <div class="content">
            """
            
            # Contenido de las pestañas
            dashboard_html += """
                    <div id="overview" class="tab-content active">
                        <h2>📋 Resumen del Análisis</h2>
                        <p>Este dashboard presenta un análisis estadístico completo de las características balísticas.</p>
                        <ul>
            """
            
            if 'correlation' in analysis_results:
                dashboard_html += '<li><strong>Análisis de Correlaciones:</strong> Identifica relaciones entre características</li>'
            if 'clustering' in analysis_results:
                dashboard_html += '<li><strong>Análisis de Clustering:</strong> Agrupa muestras similares</li>'
            if 'pca' in analysis_results:
                dashboard_html += '<li><strong>Análisis PCA:</strong> Reduce dimensionalidad y identifica componentes principales</li>'
            
            dashboard_html += """
                        </ul>
                        <p>Utilice las pestañas superiores para navegar entre los diferentes análisis.</p>
                    </div>
            """
            
            # Generar visualizaciones individuales y agregarlas al dashboard
            temp_files = []
            
            if 'correlation' in analysis_results:
                corr_file = self.create_interactive_correlation_heatmap(
                    analysis_results['correlation'], 
                    os.path.join(self.output_dir, "temp_correlation.html")
                )
                temp_files.append(corr_file)
                
                # Leer contenido del archivo de correlación
                with open(corr_file, 'r', encoding='utf-8') as f:
                    corr_content = f.read()
                    # Extraer solo el div de Plotly
                    start = corr_content.find('<div id="')
                    end = corr_content.find('</script>', start) + 9
                    if start != -1 and end != -1:
                        corr_plot = corr_content[start:end]
                        dashboard_html += f"""
                            <div id="correlation" class="tab-content">
                                <h2>🔗 Análisis de Correlaciones</h2>
                                {corr_plot}
                            </div>
                        """
            
            if 'clustering' in analysis_results:
                cluster_file = self.create_interactive_clustering_plot(
                    analysis_results['clustering'], 
                    features_data,
                    os.path.join(self.output_dir, "temp_clustering.html")
                )
                temp_files.append(cluster_file)
                
                with open(cluster_file, 'r', encoding='utf-8') as f:
                    cluster_content = f.read()
                    start = cluster_content.find('<div id="')
                    end = cluster_content.find('</script>', start) + 9
                    if start != -1 and end != -1:
                        cluster_plot = cluster_content[start:end]
                        dashboard_html += f"""
                            <div id="clustering" class="tab-content">
                                <h2>🎯 Análisis de Clustering</h2>
                                {cluster_plot}
                            </div>
                        """
            
            if 'pca' in analysis_results:
                pca_file = self.create_interactive_pca_plot(
                    analysis_results['pca'], 
                    features_data,
                    os.path.join(self.output_dir, "temp_pca.html")
                )
                temp_files.append(pca_file)
                
                with open(pca_file, 'r', encoding='utf-8') as f:
                    pca_content = f.read()
                    start = pca_content.find('<div id="')
                    end = pca_content.find('</script>', start) + 9
                    if start != -1 and end != -1:
                        pca_plot = pca_content[start:end]
                        dashboard_html += f"""
                            <div id="pca" class="tab-content">
                                <h2>📈 Análisis de Componentes Principales (PCA)</h2>
                                {pca_plot}
                            </div>
                        """
            
            # JavaScript para navegación de pestañas
            dashboard_html += """
                </div>
                
                <script>
                    function showTab(tabName) {
                        // Ocultar todos los contenidos
                        var contents = document.getElementsByClassName('tab-content');
                        for (var i = 0; i < contents.length; i++) {
                            contents[i].classList.remove('active');
                        }
                        
                        // Desactivar todas las pestañas
                        var tabs = document.getElementsByClassName('tab');
                        for (var i = 0; i < tabs.length; i++) {
                            tabs[i].classList.remove('active');
                        }
                        
                        // Mostrar contenido seleccionado
                        document.getElementById(tabName).classList.add('active');
                        
                        // Activar pestaña seleccionada
                        event.target.classList.add('active');
                    }
                </script>
            </body>
            </html>
            """
            
            # Guardar dashboard
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"interactive_dashboard_{timestamp}.html")
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            # Limpiar archivos temporales
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            self.logger.info(f"Dashboard interactivo guardado en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creando dashboard interactivo: {str(e)}")
            return self.generate_comprehensive_report(analysis_results, features_data)
        
        try:
            # Crear subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Varianza Explicada', 'Componentes Principales',
                              'Proyección PCA', 'Importancia de Características'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Varianza explicada
            if 'explained_variance_ratio' in pca_data:
                variance_ratio = pca_data['explained_variance_ratio']
                cumulative_variance = np.cumsum(variance_ratio)
                
                fig.add_trace(
                    go.Bar(
                        x=[f'PC{i+1}' for i in range(len(variance_ratio))],
                        y=variance_ratio,
                        name='Varianza Individual',
                        marker_color='lightblue',
                        hovertemplate='<b>%{x}</b><br>Varianza: %{y:.1%}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                        y=cumulative_variance,
                        mode='lines+markers',
                        name='Varianza Acumulada',
                        line=dict(color='red', width=2),
                        hovertemplate='<b>%{x}</b><br>Varianza Acumulada: %{y:.1%}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.update_yaxes(title_text='Varianza Explicada', row=1, col=1)
            
            # Proyección PCA
            if 'transformed_data' in pca_data and features_data:
                transformed_data = np.array(pca_data['transformed_data'])
                
                fig.add_trace(
                    go.Scatter(
                        x=transformed_data[:, 0],
                        y=transformed_data[:, 1] if transformed_data.shape[1] > 1 else np.zeros(len(transformed_data)),
                        mode='markers',
                        name='Muestras',
                        marker=dict(size=8, opacity=0.7),
                        hovertemplate='<b>Muestra %{pointNumber}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text='PC1', row=2, col=1)
                fig.update_yaxes(title_text='PC2', row=2, col=1)
            
            # Importancia de características
            if 'feature_importance' in pca_data:
                feature_importance = pca_data['feature_importance']
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                fig.add_trace(
                    go.Bar(
                        x=features,
                        y=importance,
                        name='Importancia PC1',
                        marker_color='lightgreen',
                        hovertemplate='<b>%{x}</b><br>Importancia: %{y:.3f}<extra></extra>'
                    ),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text='Características', row=2, col=2)
                fig.update_yaxes(title_text='Importancia', row=2, col=2)
            
            # Configuración general
            fig.update_layout(
                title={
                    'text': 'Análisis Interactivo de PCA',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                height=800,
                showlegend=True
            )
            
            # Guardar archivo HTML
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"interactive_pca_{timestamp}.html")
            
            pyo.plot(fig, filename=save_path, auto_open=False)
            self.logger.info(f"Visualización interactiva de PCA guardada en: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creando visualización interactiva de PCA: {str(e)}")
            return self.visualize_pca_analysis(pca_data, features_data, save_path)


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