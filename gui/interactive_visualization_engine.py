#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Visualizaciones Interactivas - SIGeC-Balística
======================================================

Motor completo de visualizaciones interactivas que integra Plotly y Bokeh
para análisis balístico en tiempo real con capacidades avanzadas.

Funcionalidades:
- Gráficos interactivos 3D y 2D
- Análisis en tiempo real
- Dashboards dinámicos
- Visualizaciones de clustering avanzado
- Mapas de calor interactivos
- Análisis de correlación en vivo
- Exportación multi-formato

Autor: SIGeC-Balistica Team
Fecha: Diciembre 2024
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime
import tempfile
import base64
from io import BytesIO

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QTabWidget, QGroupBox, QCheckBox,
    QSlider, QSpinBox, QComboBox, QProgressBar, QTextEdit,
    QScrollArea, QGridLayout, QFormLayout, QMessageBox,
    QFileDialog, QApplication, QSizePolicy, QToolBar, QAction
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, pyqtSlot, QObject
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette
from PyQt5.QtWebEngineWidgets import QWebEngineView

# Importaciones condicionales para librerías de visualización
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    import plotly.io as pio
    from plotly.graph_objs import Figure
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.layouts import column, row, gridplot
    from bokeh.io import curdoc
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure as MPLFigure
    
    # Importaciones de visualización - Comentadas temporalmente para pruebas
    import seaborn as sns
    
    # Mock temporal para seaborn
    class MockSeaborn:
        def set_style(self, *args, **kwargs):
            pass
        def heatmap(self, *args, **kwargs):
            pass
        def scatterplot(self, *args, **kwargs):
            pass
    
    sns = MockSeaborn()
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class InteractiveVisualizationConfig:
    """Configuración para visualizaciones interactivas"""
    
    def __init__(self):
        self.theme = "plotly_white"
        self.color_palette = px.colors.qualitative.Set3
        self.default_width = 800
        self.default_height = 600
        self.animation_duration = 500
        self.update_interval = 100  # ms
        self.enable_3d = True
        self.enable_animations = True
        self.auto_refresh = True
        self.export_formats = ['html', 'png', 'svg', 'pdf']
        
    def to_dict(self) -> Dict:
        """Convierte la configuración a diccionario"""
        return {
            'theme': self.theme,
            'color_palette': self.color_palette,
            'default_width': self.default_width,
            'default_height': self.default_height,
            'animation_duration': self.animation_duration,
            'update_interval': self.update_interval,
            'enable_3d': self.enable_3d,
            'enable_animations': self.enable_animations,
            'auto_refresh': self.auto_refresh,
            'export_formats': self.export_formats
        }

class RealTimeDataProcessor(QThread):
    """Procesador de datos en tiempo real para visualizaciones"""
    
    dataUpdated = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, data_source: Callable = None):
        super().__init__()
        self.data_source = data_source
        self.running = False
        self.update_interval = 1000  # ms
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_data)
        
    def start_processing(self):
        """Inicia el procesamiento en tiempo real"""
        self.running = True
        self.timer.start(self.update_interval)
        
    def stop_processing(self):
        """Detiene el procesamiento"""
        self.running = False
        self.timer.stop()
        
    def set_update_interval(self, interval_ms: int):
        """Establece el intervalo de actualización"""
        self.update_interval = interval_ms
        if self.timer.isActive():
            self.timer.setInterval(interval_ms)
            
    @pyqtSlot()
    def process_data(self):
        """Procesa los datos y emite señal de actualización"""
        try:
            if self.data_source and self.running:
                data = self.data_source()
                if data:
                    self.dataUpdated.emit(data)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class PlotlyVisualizationEngine:
    """Motor de visualizaciones con Plotly"""
    
    def __init__(self, config: InteractiveVisualizationConfig):
        self.config = config
        self.figures = {}
        
    def create_3d_scatter_plot(self, data: Dict, title: str = "Análisis 3D") -> str:
        """Crea un gráfico de dispersión 3D interactivo"""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_message("Plotly no disponible")
            
        try:
            fig = go.Figure(data=go.Scatter3d(
                x=data.get('x', []),
                y=data.get('y', []),
                z=data.get('z', []),
                mode='markers',
                marker=dict(
                    size=data.get('sizes', 5),
                    color=data.get('colors', self.config.color_palette[0]),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Intensidad")
                ),
                text=data.get('labels', []),
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x}<br>' +
                             'Y: %{y}<br>' +
                             'Z: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=data.get('x_label', 'X'),
                    yaxis_title=data.get('y_label', 'Y'),
                    zaxis_title=data.get('z_label', 'Z'),
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=1.2)
                    )
                ),
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.theme
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creando gráfico 3D: {e}")
            return self._create_error_message(str(e))
    
    def create_interactive_heatmap(self, data: np.ndarray, 
                                 x_labels: List = None, 
                                 y_labels: List = None,
                                 title: str = "Mapa de Calor Interactivo") -> str:
        """Crea un mapa de calor interactivo"""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_message("Plotly no disponible")
            
        try:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=x_labels or list(range(data.shape[1])),
                y=y_labels or list(range(data.shape[0])),
                colorscale='RdYlBu_r',
                hoverongaps=False,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Valor: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Características",
                yaxis_title="Muestras",
                width=self.config.default_width,
                height=self.config.default_height,
                template=self.config.theme
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creando mapa de calor: {e}")
            return self._create_error_message(str(e))
    
    def create_clustering_visualization(self, data: Dict, 
                                     algorithm: str = "kmeans",
                                     title: str = "Análisis de Clustering") -> str:
        """Crea visualización interactiva de clustering avanzado"""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_message("Plotly no disponible")
            
        try:
            # Determinar el número de subplots basado en los datos disponibles
            has_pca = 'pca_x' in data and 'pca_y' in data
            has_3d = 'z' in data
            has_comparative = 'comparative_analysis' in data
            
            if has_comparative:
                # Layout para análisis comparativo
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=('Clustering Principal', 'Distribución de Clusters', 
                                  'Métricas de Calidad', 'Análisis PCA/3D',
                                  'Comparación de Métodos', 'Ranking de Algoritmos'),
                    specs=[[{"type": "scatter3d" if has_3d else "scatter"}, {"type": "bar"}],
                           [{"type": "bar"}, {"type": "scatter"}],
                           [{"type": "heatmap"}, {"type": "bar"}]]
                )
            else:
                # Layout estándar
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Clustering 2D/3D', 'Distribución de Clusters', 
                                  'Métricas de Calidad', 'Análisis PCA'),
                    specs=[[{"type": "scatter3d" if has_3d else "scatter"}, {"type": "bar"}],
                           [{"type": "bar"}, {"type": "scatter"}]]
                )
            
            # Gráfico principal de clustering (2D o 3D)
            colors = data.get('cluster_colors', self.config.color_palette)
            labels = data.get('labels', [])
            
            if has_3d:
                # Visualización 3D
                fig.add_trace(
                    go.Scatter3d(
                        x=data.get('x', []),
                        y=data.get('y', []),
                        z=data.get('z', []),
                        mode='markers',
                        marker=dict(
                            color=labels,
                            colorscale='Set3',
                            size=6,
                            line=dict(width=1, color='white'),
                            opacity=0.8
                        ),
                        text=data.get('point_labels', []),
                        name='Puntos 3D',
                        hovertemplate='Cluster: %{marker.color}<br>' +
                                     'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Agregar centroides si están disponibles
                if 'centroids' in data:
                    centroids = np.array(data['centroids'])
                    if centroids.shape[1] >= 3:
                        fig.add_trace(
                            go.Scatter3d(
                                x=centroids[:, 0],
                                y=centroids[:, 1],
                                z=centroids[:, 2],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=12,
                                    symbol='diamond',
                                    line=dict(width=2, color='black')
                                ),
                                name='Centroides',
                                hovertemplate='Centroide<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
                            ),
                            row=1, col=1
                        )
            else:
                # Visualización 2D
                fig.add_trace(
                    go.Scatter(
                        x=data.get('x', []),
                        y=data.get('y', []),
                        mode='markers',
                        marker=dict(
                            color=labels,
                            colorscale='Set3',
                            size=8,
                            line=dict(width=1, color='white')
                        ),
                        text=data.get('point_labels', []),
                        name='Puntos',
                        hovertemplate='Cluster: %{marker.color}<br>' +
                                     'X: %{x}<br>Y: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Agregar centroides 2D
                if 'centroids' in data:
                    centroids = np.array(data['centroids'])
                    fig.add_trace(
                        go.Scatter(
                            x=centroids[:, 0],
                            y=centroids[:, 1],
                            mode='markers',
                            marker=dict(
                                color='red',
                                size=15,
                                symbol='diamond',
                                line=dict(width=2, color='black')
                            ),
                            name='Centroides',
                            hovertemplate='Centroide<br>X: %{x}<br>Y: %{y}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            # Distribución de clusters
            cluster_counts = data.get('cluster_counts', [])
            if cluster_counts:
                fig.add_trace(
                    go.Bar(
                        x=[f'Cluster {i}' for i in range(len(cluster_counts))],
                        y=cluster_counts,
                        name='Tamaño de Clusters',
                        marker_color=colors[:len(cluster_counts)],
                        hovertemplate='%{x}<br>Tamaño: %{y}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # Métricas de calidad
            metrics = data.get('metrics', {})
            if metrics:
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                
                fig.add_trace(
                    go.Bar(
                        x=metric_names,
                        y=metric_values,
                        name='Métricas',
                        marker_color='lightblue',
                        hovertemplate='%{x}<br>Valor: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Análisis PCA o visualización adicional
            if has_pca:
                fig.add_trace(
                    go.Scatter(
                        x=data['pca_x'],
                        y=data['pca_y'],
                        mode='markers',
                        marker=dict(
                            color=labels,
                            colorscale='Set3',
                            size=6
                        ),
                        name='PCA',
                        hovertemplate='PC1: %{x}<br>PC2: %{y}<extra></extra>'
                    ),
                    row=2, col=2
                )
            elif 'reachability_distances' in data and algorithm.lower() == 'optics':
                # Gráfico de reachability para OPTICS
                reach_distances = data['reachability_distances']
                ordering = data.get('ordering', list(range(len(reach_distances))))
                
                fig.add_trace(
                    go.Scatter(
                        x=ordering,
                        y=reach_distances,
                        mode='lines+markers',
                        name='Reachability Plot',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4),
                        hovertemplate='Punto: %{x}<br>Distancia: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # Análisis comparativo si está disponible
            if has_comparative:
                comp_data = data['comparative_analysis']
                
                # Matriz de similitud entre métodos
                if 'similarity_matrix' in comp_data:
                    sim_data = comp_data['similarity_matrix']
                    fig.add_trace(
                        go.Heatmap(
                            z=sim_data['matrix'],
                            x=sim_data['methods'],
                            y=sim_data['methods'],
                            colorscale='RdYlBu',
                            name='Similitud',
                            hovertemplate='%{x} vs %{y}<br>ARI: %{z:.4f}<extra></extra>'
                        ),
                        row=3, col=1
                    )
                
                # Ranking de métodos
                if 'ranking' in comp_data:
                    ranking = comp_data['ranking']
                    methods = [r['method'] for r in ranking]
                    scores = [r['composite_score'] for r in ranking]
                    
                    fig.add_trace(
                        go.Bar(
                            x=methods,
                            y=scores,
                            name='Ranking',
                            marker_color='green',
                            hovertemplate='%{x}<br>Score: %{y:.4f}<extra></extra>'
                        ),
                        row=3, col=2
                    )
            
            # Configurar layout
            layout_config = {
                'title': f"{title} - {algorithm.upper()}",
                'showlegend': True,
                'width': self.config.default_width * 1.5,
                'height': self.config.default_height * (1.5 if has_comparative else 1.2),
                'template': self.config.theme
            }
            
            # Configuraciones específicas para 3D
            if has_3d:
                layout_config['scene'] = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            
            fig.update_layout(**layout_config)
            
            # Agregar anotaciones con información del algoritmo
            algorithm_info = self._get_algorithm_info(algorithm, data)
            if algorithm_info:
                fig.add_annotation(
                    text=algorithm_info,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creando visualización de clustering: {e}")
            return self._create_error_message(str(e))
    
    def _get_algorithm_info(self, algorithm: str, data: Dict) -> str:
        """Genera información contextual del algoritmo"""
        info_lines = [f"Algoritmo: {algorithm.upper()}"]
        
        if 'n_clusters' in data:
            info_lines.append(f"Clusters: {data['n_clusters']}")
        
        if algorithm.lower() == 'dbscan':
            if 'eps' in data:
                info_lines.append(f"Eps: {data['eps']:.4f}")
            if 'min_samples' in data:
                info_lines.append(f"Min Samples: {data['min_samples']}")
            if 'n_noise_points' in data:
                info_lines.append(f"Ruido: {data['n_noise_points']} puntos")
        
        elif algorithm.lower() == 'optics':
            if 'min_samples' in data:
                info_lines.append(f"Min Samples: {data['min_samples']}")
            if 'xi' in data:
                info_lines.append(f"Xi: {data['xi']}")
        
        elif algorithm.lower() == 'meanshift':
            if 'bandwidth' in data:
                info_lines.append(f"Bandwidth: {data['bandwidth']:.4f}")
        
        elif algorithm.lower() == 'birch':
            if 'threshold' in data:
                info_lines.append(f"Threshold: {data['threshold']}")
            if 'branching_factor' in data:
                info_lines.append(f"Branching: {data['branching_factor']}")
        
        if 'silhouette_score' in data:
            info_lines.append(f"Silhouette: {data['silhouette_score']:.4f}")
        
        return "<br>".join(info_lines)
    
    def create_real_time_dashboard(self, data_streams: Dict) -> str:
        """Crea un dashboard en tiempo real"""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_message("Plotly no disponible")
            
        try:
            # Crear dashboard con múltiples métricas
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Calidad en Tiempo Real', 'Matches Detectados',
                              'Uso de CPU/GPU', 'Memoria Utilizada',
                              'Throughput', 'Errores/Alertas'),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Métricas de calidad en tiempo real
            quality_data = data_streams.get('quality', {})
            fig.add_trace(
                go.Scatter(
                    x=quality_data.get('timestamps', []),
                    y=quality_data.get('values', []),
                    mode='lines+markers',
                    name='Calidad',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # Matches detectados
            matches_data = data_streams.get('matches', {})
            fig.add_trace(
                go.Bar(
                    x=matches_data.get('categories', []),
                    y=matches_data.get('counts', []),
                    name='Matches',
                    marker_color='blue'
                ),
                row=1, col=2
            )
            
            # Uso de recursos
            cpu_data = data_streams.get('cpu', {})
            gpu_data = data_streams.get('gpu', {})
            
            fig.add_trace(
                go.Scatter(
                    x=cpu_data.get('timestamps', []),
                    y=cpu_data.get('values', []),
                    mode='lines',
                    name='CPU %',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            if gpu_data:
                fig.add_trace(
                    go.Scatter(
                        x=gpu_data.get('timestamps', []),
                        y=gpu_data.get('values', []),
                        mode='lines',
                        name='GPU %',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
            
            # Memoria
            memory_data = data_streams.get('memory', {})
            fig.add_trace(
                go.Scatter(
                    x=memory_data.get('timestamps', []),
                    y=memory_data.get('values', []),
                    mode='lines+markers',
                    name='Memoria MB',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
            
            # Throughput
            throughput_data = data_streams.get('throughput', {})
            fig.add_trace(
                go.Scatter(
                    x=throughput_data.get('timestamps', []),
                    y=throughput_data.get('values', []),
                    mode='lines',
                    name='Imágenes/seg',
                    line=dict(color='teal')
                ),
                row=3, col=1
            )
            
            # Errores y alertas
            errors_data = data_streams.get('errors', {})
            fig.add_trace(
                go.Bar(
                    x=errors_data.get('types', []),
                    y=errors_data.get('counts', []),
                    name='Errores',
                    marker_color='red'
                ),
                row=3, col=2
            )
            
            fig.update_layout(
                title="Dashboard de Monitoreo en Tiempo Real",
                showlegend=True,
                width=self.config.default_width * 1.8,
                height=self.config.default_height * 2,
                template=self.config.theme,
                updatemenus=[{
                    'type': 'buttons',
                    'direction': 'left',
                    'buttons': [
                        {'label': 'Play', 'method': 'animate', 'args': [None]},
                        {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]}
                    ],
                    'x': 0.1,
                    'y': 1.02
                }]
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creando dashboard: {e}")
            return self._create_error_message(str(e))
    
    def _create_fallback_message(self, message: str) -> str:
        """Crea mensaje de fallback cuando no hay librerías disponibles"""
        return f"""
        <div style="padding: 20px; text-align: center; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px;">
            <h3>Visualización No Disponible</h3>
            <p>{message}</p>
            <p>Por favor, instale las dependencias necesarias para habilitar visualizaciones interactivas.</p>
        </div>
        """
    
    def _create_error_message(self, error: str) -> str:
        """Crea mensaje de error"""
        return f"""
        <div style="padding: 20px; text-align: center; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;">
            <h3>Error en Visualización</h3>
            <p>Ocurrió un error al generar la visualización:</p>
            <code>{error}</code>
        </div>
        """

class BokehVisualizationEngine:
    """Motor de visualizaciones con Bokeh"""
    
    def __init__(self, config: InteractiveVisualizationConfig):
        self.config = config
        
    def create_interactive_scatter(self, data: Dict, title: str = "Análisis Interactivo") -> str:
        """Crea gráfico de dispersión interactivo con Bokeh"""
        if not BOKEH_AVAILABLE:
            return self._create_fallback_message("Bokeh no disponible")
            
        try:
            p = figure(
                title=title,
                width=self.config.default_width,
                height=self.config.default_height,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            
            # Agregar herramienta de hover
            hover = HoverTool(tooltips=[
                ("Índice", "@index"),
                ("X", "@x"),
                ("Y", "@y"),
                ("Etiqueta", "@label")
            ])
            p.add_tools(hover)
            
            # Crear gráfico de dispersión
            p.circle(
                data.get('x', []),
                data.get('y', []),
                size=data.get('sizes', 10),
                color=data.get('colors', 'blue'),
                alpha=0.7
            )
            
            # Generar HTML
            html = file_html(p, CDN, title)
            return html
            
        except Exception as e:
            logger.error(f"Error creando gráfico Bokeh: {e}")
            return self._create_error_message(str(e))
    
    def _create_fallback_message(self, message: str) -> str:
        """Crea mensaje de fallback"""
        return f"""
        <div style="padding: 20px; text-align: center; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px;">
            <h3>Bokeh No Disponible</h3>
            <p>{message}</p>
        </div>
        """
    
    def _create_error_message(self, error: str) -> str:
        """Crea mensaje de error"""
        return f"""
        <div style="padding: 20px; text-align: center; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;">
            <h3>Error en Bokeh</h3>
            <p>{error}</p>
        </div>
        """

class InteractiveVisualizationWidget(QWidget):
    """Widget principal para visualizaciones interactivas"""
    
    visualizationUpdated = pyqtSignal(str)
    exportRequested = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = InteractiveVisualizationConfig()
        self.plotly_engine = PlotlyVisualizationEngine(self.config)
        self.bokeh_engine = BokehVisualizationEngine(self.config)
        self.real_time_processor = RealTimeDataProcessor()
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Barra de herramientas
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Área principal con pestañas
        self.tab_widget = QTabWidget()
        
        # Pestaña de visualizaciones estáticas
        static_tab = self._create_static_visualizations_tab()
        self.tab_widget.addTab(static_tab, "📊 Visualizaciones")
        
        # Pestaña de dashboard en tiempo real
        realtime_tab = self._create_realtime_dashboard_tab()
        self.tab_widget.addTab(realtime_tab, "📈 Tiempo Real")
        
        # Pestaña de clustering avanzado
        clustering_tab = self._create_clustering_tab()
        self.tab_widget.addTab(clustering_tab, "🎯 Clustering")
        
        # Pestaña de configuración
        config_tab = self._create_configuration_tab()
        self.tab_widget.addTab(config_tab, "⚙️ Configuración")
        
        layout.addWidget(self.tab_widget)
        
    def _create_toolbar(self) -> QToolBar:
        """Crea la barra de herramientas"""
        toolbar = QToolBar()
        
        # Acción de actualizar
        refresh_action = QAction("🔄 Actualizar", self)
        refresh_action.triggered.connect(self.refresh_visualizations)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        # Acción de exportar
        export_action = QAction("💾 Exportar", self)
        export_action.triggered.connect(self.export_current_visualization)
        toolbar.addAction(export_action)
        
        # Acción de pantalla completa
        fullscreen_action = QAction("🖥️ Pantalla Completa", self)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        toolbar.addAction(fullscreen_action)
        
        return toolbar
        
    def _create_static_visualizations_tab(self) -> QWidget:
        """Crea la pestaña de visualizaciones estáticas"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controles
        controls_group = QGroupBox("Controles de Visualización")
        controls_layout = QHBoxLayout(controls_group)
        
        # Selector de tipo de gráfico
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Dispersión 3D", "Mapa de Calor", "Análisis PCA",
            "Distribución de Características", "Correlación"
        ])
        controls_layout.addWidget(QLabel("Tipo:"))
        controls_layout.addWidget(self.chart_type_combo)
        
        # Botón de generar
        generate_btn = QPushButton("Generar Visualización")
        generate_btn.clicked.connect(self.generate_static_visualization)
        controls_layout.addWidget(generate_btn)
        
        layout.addWidget(controls_group)
        
        # Área de visualización
        self.static_webview = QWebEngineView()
        layout.addWidget(self.static_webview)
        
        return widget
        
    def _create_realtime_dashboard_tab(self) -> QWidget:
        """Crea la pestaña del dashboard en tiempo real"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controles de tiempo real
        controls_group = QGroupBox("Controles de Tiempo Real")
        controls_layout = QHBoxLayout(controls_group)
        
        # Botón de inicio/parada
        self.realtime_btn = QPushButton("▶️ Iniciar Monitoreo")
        self.realtime_btn.clicked.connect(self.toggle_realtime_monitoring)
        controls_layout.addWidget(self.realtime_btn)
        
        # Intervalo de actualización
        controls_layout.addWidget(QLabel("Intervalo (ms):"))
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(100, 10000)
        self.interval_spinbox.setValue(1000)
        self.interval_spinbox.valueChanged.connect(self.update_refresh_interval)
        controls_layout.addWidget(self.interval_spinbox)
        
        # Indicador de estado
        self.status_label = QLabel("⏹️ Detenido")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls_group)
        
        # Dashboard web
        self.dashboard_webview = QWebEngineView()
        layout.addWidget(self.dashboard_webview)
        
        return widget
        
    def _create_clustering_tab(self) -> QWidget:
        """Crea la pestaña de clustering avanzado"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controles de clustering
        controls_group = QGroupBox("Configuración de Clustering")
        controls_layout = QFormLayout(controls_group)
        
        # Algoritmo
        self.clustering_algo_combo = QComboBox()
        self.clustering_algo_combo.addItems([
            "K-Means", "DBSCAN", "Hierarchical", "Spectral", "GMM",
            "OPTICS", "Mean Shift", "BIRCH"
        ])
        controls_layout.addRow("Algoritmo:", self.clustering_algo_combo)
        
        # Número de clusters
        self.n_clusters_spinbox = QSpinBox()
        self.n_clusters_spinbox.setRange(2, 20)
        self.n_clusters_spinbox.setValue(3)
        controls_layout.addRow("N° Clusters:", self.n_clusters_spinbox)
        
        # Botón de análisis
        analyze_btn = QPushButton("Analizar Clustering")
        analyze_btn.clicked.connect(self.analyze_clustering)
        controls_layout.addRow(analyze_btn)
        
        layout.addWidget(controls_group)
        
        # Visualización de clustering
        self.clustering_webview = QWebEngineView()
        layout.addWidget(self.clustering_webview)
        
        return widget
        
    def _create_configuration_tab(self) -> QWidget:
        """Crea la pestaña de configuración"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configuración de tema
        theme_group = QGroupBox("Configuración Visual")
        theme_layout = QFormLayout(theme_group)
        
        # Tema
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([
            "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"
        ])
        self.theme_combo.setCurrentText(self.config.theme)
        self.theme_combo.currentTextChanged.connect(self.update_theme)
        theme_layout.addRow("Tema:", self.theme_combo)
        
        # Dimensiones por defecto
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(400, 2000)
        self.width_spinbox.setValue(self.config.default_width)
        self.width_spinbox.valueChanged.connect(self.update_default_width)
        theme_layout.addRow("Ancho:", self.width_spinbox)
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(300, 1500)
        self.height_spinbox.setValue(self.config.default_height)
        self.height_spinbox.valueChanged.connect(self.update_default_height)
        theme_layout.addRow("Alto:", self.height_spinbox)
        
        # Opciones avanzadas
        self.enable_3d_cb = QCheckBox("Habilitar gráficos 3D")
        self.enable_3d_cb.setChecked(self.config.enable_3d)
        self.enable_3d_cb.toggled.connect(self.toggle_3d_support)
        theme_layout.addRow(self.enable_3d_cb)
        
        self.enable_animations_cb = QCheckBox("Habilitar animaciones")
        self.enable_animations_cb.setChecked(self.config.enable_animations)
        self.enable_animations_cb.toggled.connect(self.toggle_animations)
        theme_layout.addRow(self.enable_animations_cb)
        
        layout.addWidget(theme_group)
        
        # Información del sistema
        info_group = QGroupBox("Información del Sistema")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(200)
        
        system_info = f"""
Librerías Disponibles:
• Plotly: {'✅ Disponible' if PLOTLY_AVAILABLE else '❌ No disponible'}
• Bokeh: {'✅ Disponible' if BOKEH_AVAILABLE else '❌ No disponible'}
• Matplotlib: {'✅ Disponible' if MATPLOTLIB_AVAILABLE else '❌ No disponible'}

Configuración Actual:
• Tema: {self.config.theme}
• Dimensiones: {self.config.default_width}x{self.config.default_height}
• Gráficos 3D: {'Habilitado' if self.config.enable_3d else 'Deshabilitado'}
• Animaciones: {'Habilitadas' if self.config.enable_animations else 'Deshabilitadas'}
        """
        
        info_text.setPlainText(system_info.strip())
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        return widget
        
    def _connect_signals(self):
        """Conecta las señales"""
        self.real_time_processor.dataUpdated.connect(self.update_realtime_dashboard)
        self.real_time_processor.errorOccurred.connect(self.handle_realtime_error)
        
    # Métodos de funcionalidad
    
    def generate_static_visualization(self):
        """Genera una visualización estática"""
        chart_type = self.chart_type_combo.currentText()
        
        # Datos de ejemplo (en implementación real vendrían del análisis)
        sample_data = self._generate_sample_data(chart_type)
        
        try:
            if chart_type == "Dispersión 3D":
                html = self.plotly_engine.create_3d_scatter_plot(sample_data, "Análisis 3D de Características")
            elif chart_type == "Mapa de Calor":
                correlation_matrix = np.random.rand(10, 10)
                html = self.plotly_engine.create_interactive_heatmap(
                    correlation_matrix, 
                    title="Correlación de Características"
                )
            else:
                html = self.plotly_engine.create_3d_scatter_plot(sample_data, chart_type)
                
            self.static_webview.setHtml(html)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generando visualización: {e}")
    
    def toggle_realtime_monitoring(self):
        """Alterna el monitoreo en tiempo real"""
        if not self.real_time_processor.running:
            self.real_time_processor.data_source = self._generate_realtime_data
            self.real_time_processor.start_processing()
            self.realtime_btn.setText("⏸️ Pausar Monitoreo")
            self.status_label.setText("▶️ Ejecutándose")
        else:
            self.real_time_processor.stop_processing()
            self.realtime_btn.setText("▶️ Iniciar Monitoreo")
            self.status_label.setText("⏹️ Detenido")
    
    def analyze_clustering(self):
        """Analiza clustering con el algoritmo seleccionado"""
        algorithm = self.clustering_algo_combo.currentText().lower()
        n_clusters = self.n_clusters_spinbox.value()
        
        # Generar datos de clustering de ejemplo
        clustering_data = self._generate_clustering_data(algorithm, n_clusters)
        
        try:
            html = self.plotly_engine.create_clustering_visualization(
                clustering_data, 
                algorithm, 
                f"Análisis de Clustering - {algorithm.upper()}"
            )
            self.clustering_webview.setHtml(html)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error en análisis de clustering: {e}")
    
    def refresh_visualizations(self):
        """Actualiza todas las visualizaciones"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Visualizaciones estáticas
            self.generate_static_visualization()
        elif current_tab == 1:  # Dashboard tiempo real
            if self.real_time_processor.running:
                self.update_realtime_dashboard(self._generate_realtime_data())
        elif current_tab == 2:  # Clustering
            self.analyze_clustering()
    
    def export_current_visualization(self):
        """Exporta la visualización actual"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Exportar Visualización",
            f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html);;PNG Files (*.png);;SVG Files (*.svg)"
        )
        
        if filename:
            try:
                current_tab = self.tab_widget.currentIndex()
                if current_tab == 0:
                    webview = self.static_webview
                elif current_tab == 1:
                    webview = self.dashboard_webview
                elif current_tab == 2:
                    webview = self.clustering_webview
                else:
                    return
                
                # Exportar HTML
                if filename.endswith('.html'):
                    webview.page().toHtml(lambda html: self._save_html(html, filename))
                else:
                    QMessageBox.information(self, "Exportación", "Formato no soportado aún")
                    
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error exportando: {e}")
    
    def toggle_fullscreen(self):
        """Alterna modo pantalla completa"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    # Métodos de configuración
    
    def update_theme(self, theme: str):
        """Actualiza el tema"""
        self.config.theme = theme
        self.refresh_visualizations()
    
    def update_default_width(self, width: int):
        """Actualiza el ancho por defecto"""
        self.config.default_width = width
    
    def update_default_height(self, height: int):
        """Actualiza el alto por defecto"""
        self.config.default_height = height
    
    def toggle_3d_support(self, enabled: bool):
        """Alterna soporte 3D"""
        self.config.enable_3d = enabled
    
    def toggle_animations(self, enabled: bool):
        """Alterna animaciones"""
        self.config.enable_animations = enabled
    
    def update_refresh_interval(self, interval: int):
        """Actualiza el intervalo de actualización"""
        self.real_time_processor.set_update_interval(interval)
    
    # Métodos de datos de ejemplo
    
    def _generate_sample_data(self, chart_type: str) -> Dict:
        """Genera datos de ejemplo para visualizaciones"""
        np.random.seed(42)
        n_points = 100
        
        return {
            'x': np.random.randn(n_points),
            'y': np.random.randn(n_points),
            'z': np.random.randn(n_points),
            'colors': np.random.randint(0, 5, n_points),
            'sizes': np.random.randint(5, 15, n_points),
            'labels': [f'Punto {i}' for i in range(n_points)],
            'x_label': 'Característica X',
            'y_label': 'Característica Y',
            'z_label': 'Característica Z'
        }
    
    def _generate_clustering_data(self, algorithm: str, n_clusters: int) -> Dict:
        """Genera datos de clustering de ejemplo"""
        np.random.seed(42)
        n_points = 200
        
        # Generar clusters sintéticos con diferentes patrones según el algoritmo
        if algorithm.lower() in ['optics', 'mean shift']:
            # Para OPTICS y Mean Shift, generar datos con densidades variables
            cluster_centers = np.random.randn(n_clusters, 2) * 3
            labels = np.random.randint(0, n_clusters, n_points)
            
            x = []
            y = []
            for i in range(n_points):
                center = cluster_centers[labels[i]]
                # Densidad variable para simular OPTICS
                density_factor = np.random.uniform(0.3, 1.0)
                x.append(center[0] + np.random.randn() * density_factor)
                y.append(center[1] + np.random.randn() * density_factor)
                
        elif algorithm.lower() == 'birch':
            # Para BIRCH, generar datos más compactos
            cluster_centers = np.random.randn(n_clusters, 2) * 2
            labels = np.random.randint(0, n_clusters, n_points)
            
            x = []
            y = []
            for i in range(n_points):
                center = cluster_centers[labels[i]]
                x.append(center[0] + np.random.randn() * 0.3)
                y.append(center[1] + np.random.randn() * 0.3)
        else:
            # Para algoritmos tradicionales
            cluster_centers = np.random.randn(n_clusters, 2) * 3
            labels = np.random.randint(0, n_clusters, n_points)
            
            x = []
            y = []
            for i in range(n_points):
                center = cluster_centers[labels[i]]
                x.append(center[0] + np.random.randn() * 0.5)
                y.append(center[1] + np.random.randn() * 0.5)
        
        # PCA simulado
        pca_x = np.array(x) * 0.8 + np.array(y) * 0.2
        pca_y = np.array(x) * 0.2 + np.array(y) * 0.8
        
        # Métricas específicas por algoritmo
        metrics = {
            'Silhouette Score': np.random.uniform(0.3, 0.8),
            'Calinski-Harabasz': np.random.uniform(100, 500),
            'Davies-Bouldin': np.random.uniform(0.5, 2.0)
        }
        
        # Agregar métricas específicas para algoritmos avanzados
        if algorithm.lower() == 'optics':
            metrics['Reachability Distance'] = np.random.uniform(0.1, 2.0)
        elif algorithm.lower() == 'mean shift':
            metrics['Bandwidth'] = np.random.uniform(0.5, 2.0)
        elif algorithm.lower() == 'birch':
            metrics['Threshold'] = np.random.uniform(0.1, 1.0)
            metrics['Branching Factor'] = np.random.randint(20, 100)
        
        return {
            'x': x,
            'y': y,
            'labels': labels,
            'cluster_counts': [np.sum(labels == i) for i in range(n_clusters)],
            'cluster_colors': self.config.color_palette[:n_clusters],
            'point_labels': [f'Muestra {i}' for i in range(n_points)],
            'pca_x': pca_x,
            'pca_y': pca_y,
            'metrics': metrics,
            'algorithm': algorithm
        }
    
    def _generate_realtime_data(self) -> Dict:
        """Genera datos en tiempo real simulados"""
        current_time = datetime.now()
        
        return {
            'quality': {
                'timestamps': [current_time],
                'values': [np.random.uniform(70, 95)]
            },
            'matches': {
                'categories': ['Exactos', 'Probables', 'Posibles'],
                'counts': [np.random.randint(0, 10), np.random.randint(0, 15), np.random.randint(0, 20)]
            },
            'cpu': {
                'timestamps': [current_time],
                'values': [np.random.uniform(20, 80)]
            },
            'gpu': {
                'timestamps': [current_time],
                'values': [np.random.uniform(10, 60)]
            },
            'memory': {
                'timestamps': [current_time],
                'values': [np.random.uniform(1000, 4000)]
            },
            'throughput': {
                'timestamps': [current_time],
                'values': [np.random.uniform(5, 25)]
            },
            'errors': {
                'types': ['Críticos', 'Advertencias', 'Info'],
                'counts': [np.random.randint(0, 3), np.random.randint(0, 8), np.random.randint(0, 15)]
            }
        }
    
    @pyqtSlot(dict)
    def update_realtime_dashboard(self, data: Dict):
        """Actualiza el dashboard en tiempo real"""
        try:
            html = self.plotly_engine.create_real_time_dashboard(data)
            self.dashboard_webview.setHtml(html)
        except Exception as e:
            logger.error(f"Error actualizando dashboard: {e}")
    
    @pyqtSlot(str)
    def handle_realtime_error(self, error: str):
        """Maneja errores del procesamiento en tiempo real"""
        logger.error(f"Error en tiempo real: {error}")
        self.status_label.setText(f"❌ Error: {error}")
    
    def _save_html(self, html: str, filename: str):
        """Guarda HTML a archivo"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html)
            QMessageBox.information(self, "Exportación", f"Visualización exportada a {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error guardando archivo: {e}")

# Función de utilidad para integración
def create_interactive_visualization_widget(parent=None) -> InteractiveVisualizationWidget:
    """Crea y retorna un widget de visualizaciones interactivas"""
    return InteractiveVisualizationWidget(parent)

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Crear y mostrar el widget
    widget = create_interactive_visualization_widget()
    widget.setWindowTitle("Motor de Visualizaciones Interactivas - SIGeC-Balística")
    widget.resize(1200, 800)
    widget.show()
    
    sys.exit(app.exec_())