#!/usr/bin/env python3
"""
Dashboard en Tiempo Real para Sistema de Monitoreo SIGeC-Balistica
Dashboard interactivo con visualizaciones en tiempo real de métricas del sistema

Este módulo implementa:
- Dashboard web interactivo con Plotly y Dash
- Visualizaciones en tiempo real de métricas del sistema
- Alertas visuales y notificaciones
- Gráficos históricos y de tendencias
- Panel de control de configuración
- Exportación de reportes

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import webbrowser

# Importaciones para dashboard web
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Importaciones PyQt5 para dashboard nativo
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
        QProgressBar, QPushButton, QTabWidget, QTextEdit, QScrollArea,
        QFrame, QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QApplication, QMainWindow
    )
    from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
    from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QPainter
    from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

# Importaciones internas
try:
    from performance.enhanced_monitoring_system import (
        get_enhanced_monitoring_system, EnhancedMonitoringSystem,
        MonitoringConfig, AdvancedMetric, SmartAlert, AlertSeverity
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardConfig:
    """Configuración del dashboard."""
    
    def __init__(self):
        self.web_port = 8050
        self.web_host = "get_config_value('api.host', '127.0.0.1')"
        self.update_interval_ms = 2000
        self.max_data_points = 100
        self.enable_web_dashboard = True
        self.enable_native_dashboard = True
        self.auto_refresh = True
        self.show_predictions = True
        self.show_anomalies = True
        self.theme = "dark"  # "dark" o "light"

class WebDashboard:
    """Dashboard web interactivo usando Dash y Plotly."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.WebDashboard")
        
        if not DASH_AVAILABLE:
            self.logger.error("Dash no está disponible. Instalar con: pip install dash plotly")
            return
        
        if not MONITORING_AVAILABLE:
            self.logger.error("Sistema de monitoreo no disponible")
            return
        
        self.monitoring_system = get_enhanced_monitoring_system()
        self.app = None
        self.server_thread = None
        self.running = False
        
        # Datos para gráficos
        self.metric_data = {}
        self.alert_data = []
        
        self._setup_app()
    
    def _setup_app(self):
        """Configura la aplicación Dash."""
        
        self.app = dash.Dash(__name__)
        self.app.title = "SIGeC-Balistica - Dashboard de Monitoreo"
        
        # Configurar tema
        if self.config.theme == "dark":
            self.app.layout = self._create_dark_layout()
        else:
            self.app.layout = self._create_light_layout()
        
        # Configurar callbacks
        self._setup_callbacks()
    
    def _create_dark_layout(self):
        """Crea el layout con tema oscuro."""
        
        return html.Div([
            # Header
            html.Div([
                html.H1("SIGeC-Balistica - Dashboard de Monitoreo", 
                       className="dashboard-title"),
                html.Div(id="system-status", className="system-status"),
                html.Div(id="last-update", className="last-update")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Intervalo de actualización (ms):"),
                    dcc.Input(
                        id="update-interval",
                        type="number",
                        value=self.config.update_interval_ms,
                        min=1000,
                        max=30000,
                        step=1000
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Puntos de datos máximos:"),
                    dcc.Input(
                        id="max-data-points",
                        type="number",
                        value=self.config.max_data_points,
                        min=50,
                        max=500,
                        step=10
                    )
                ], className="control-item"),
                
                html.Div([
                    dcc.Checklist(
                        id="display-options",
                        options=[
                            {"label": "Auto-refresh", "value": "auto_refresh"},
                            {"label": "Mostrar predicciones", "value": "predictions"},
                            {"label": "Mostrar anomalías", "value": "anomalies"}
                        ],
                        value=["auto_refresh", "predictions", "anomalies"]
                    )
                ], className="control-item")
            ], className="control-panel"),
            
            # Tabs
            dcc.Tabs(id="main-tabs", value="overview", children=[
                dcc.Tab(label="Vista General", value="overview"),
                dcc.Tab(label="Métricas del Sistema", value="system"),
                dcc.Tab(label="Rendimiento", value="performance"),
                dcc.Tab(label="Alertas", value="alerts"),
                dcc.Tab(label="Análisis", value="analysis")
            ]),
            
            # Content
            html.Div(id="tab-content"),
            
            # Auto-refresh component
            dcc.Interval(
                id="interval-component",
                interval=self.config.update_interval_ms,
                n_intervals=0
            ),
            
            # Store components for data
            dcc.Store(id="metrics-store"),
            dcc.Store(id="alerts-store")
            
        ], className="dashboard-container dark-theme")
    
    def _create_light_layout(self):
        """Crea el layout con tema claro."""
        # Similar al tema oscuro pero con clases CSS diferentes
        layout = self._create_dark_layout()
        # Cambiar clase del contenedor principal
        layout.className = "dashboard-container light-theme"
        return layout
    
    def _setup_callbacks(self):
        """Configura los callbacks de Dash."""
        
        @self.app.callback(
            [Output("metrics-store", "data"),
             Output("alerts-store", "data"),
             Output("system-status", "children"),
             Output("last-update", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_data(n):
            """Actualiza los datos del dashboard."""
            
            try:
                # Obtener datos del sistema de monitoreo
                dashboard_data = self.monitoring_system.get_dashboard_data()
                
                # Preparar datos de métricas
                metrics_data = {}
                for metric_name, metric_info in dashboard_data.get("current_metrics", {}).items():
                    if metric_name not in self.metric_data:
                        self.metric_data[metric_name] = {"timestamps": [], "values": []}
                    
                    # Añadir nuevo punto de datos
                    self.metric_data[metric_name]["timestamps"].append(metric_info["timestamp"])
                    self.metric_data[metric_name]["values"].append(metric_info["value"])
                    
                    # Limitar puntos de datos
                    if len(self.metric_data[metric_name]["values"]) > self.config.max_data_points:
                        self.metric_data[metric_name]["timestamps"] = self.metric_data[metric_name]["timestamps"][-self.config.max_data_points:]
                        self.metric_data[metric_name]["values"] = self.metric_data[metric_name]["values"][-self.config.max_data_points:]
                    
                    metrics_data[metric_name] = self.metric_data[metric_name].copy()
                
                # Preparar datos de alertas
                alerts_data = dashboard_data.get("active_alerts", [])
                
                # Estado del sistema
                system_status = dashboard_data.get("system_status", "unknown")
                status_colors = {
                    "healthy": "green",
                    "caution": "yellow", 
                    "warning": "orange",
                    "critical": "red"
                }
                
                status_component = html.Div([
                    html.Span("●", style={"color": status_colors.get(system_status, "gray"), "font-size": "20px"}),
                    html.Span(f" Sistema: {system_status.upper()}", style={"margin-left": "10px"})
                ])
                
                # Última actualización
                last_update = f"Última actualización: {datetime.now().strftime('%H:%M:%S')}"
                
                return metrics_data, alerts_data, status_component, last_update
                
            except Exception as e:
                self.logger.error(f"Error actualizando datos del dashboard: {e}")
                return {}, [], html.Div("Error"), "Error de actualización"
        
        @self.app.callback(
            Output("tab-content", "children"),
            [Input("main-tabs", "value"),
             Input("metrics-store", "data"),
             Input("alerts-store", "data")]
        )
        def render_tab_content(active_tab, metrics_data, alerts_data):
            """Renderiza el contenido de las pestañas."""
            
            if active_tab == "overview":
                return self._create_overview_tab(metrics_data, alerts_data)
            elif active_tab == "system":
                return self._create_system_tab(metrics_data)
            elif active_tab == "performance":
                return self._create_performance_tab(metrics_data)
            elif active_tab == "alerts":
                return self._create_alerts_tab(alerts_data)
            elif active_tab == "analysis":
                return self._create_analysis_tab(metrics_data)
            
            return html.Div("Seleccione una pestaña")
    
    def _create_overview_tab(self, metrics_data, alerts_data):
        """Crea la pestaña de vista general."""
        
        # Métricas clave
        key_metrics = ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent", "gpu_memory_usage_percent"]
        
        # Crear gráficos de gauge para métricas clave
        gauges = []
        for metric in key_metrics:
            if metric in metrics_data and metrics_data[metric]["values"]:
                current_value = metrics_data[metric]["values"][-1]
                
                gauge = dcc.Graph(
                    figure=go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=current_value,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": metric.replace("_", " ").title()},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "yellow"},
                                {"range": [80, 100], "color": "red"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90
                            }
                        }
                    )).update_layout(
                        height=300,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    ),
                    className="gauge-chart"
                )
                gauges.append(gauge)
        
        # Resumen de alertas
        alert_summary = html.Div([
            html.H3("Resumen de Alertas"),
            html.Div([
                html.Div(f"Alertas Activas: {len(alerts_data)}", className="alert-count"),
                html.Div([
                    html.Div(f"Críticas: {len([a for a in alerts_data if a.get('severity') == 'critical'])}", 
                            className="alert-critical"),
                    html.Div(f"Altas: {len([a for a in alerts_data if a.get('severity') == 'high'])}", 
                            className="alert-high"),
                    html.Div(f"Medias: {len([a for a in alerts_data if a.get('severity') == 'medium'])}", 
                            className="alert-medium")
                ], className="alert-breakdown")
            ])
        ], className="alert-summary")
        
        return html.Div([
            html.H2("Vista General del Sistema"),
            
            # Métricas clave en grid
            html.Div(gauges, className="metrics-grid"),
            
            # Resumen de alertas
            alert_summary,
            
            # Gráfico de tendencias
            html.Div([
                html.H3("Tendencias del Sistema"),
                dcc.Graph(
                    id="overview-trends",
                    figure=self._create_trends_chart(metrics_data, key_metrics)
                )
            ])
        ])
    
    def _create_system_tab(self, metrics_data):
        """Crea la pestaña de métricas del sistema."""
        
        system_metrics = [
            "cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
            "load_average_1m", "process_count", "network_sent_mb_per_sec"
        ]
        
        charts = []
        for metric in system_metrics:
            if metric in metrics_data and metrics_data[metric]["values"]:
                chart = dcc.Graph(
                    figure=self._create_time_series_chart(
                        metrics_data[metric]["timestamps"],
                        metrics_data[metric]["values"],
                        metric.replace("_", " ").title()
                    )
                )
                charts.append(chart)
        
        return html.Div([
            html.H2("Métricas del Sistema"),
            html.Div(charts, className="charts-container")
        ])
    
    def _create_performance_tab(self, metrics_data):
        """Crea la pestaña de rendimiento."""
        
        performance_metrics = [
            "app_cpu_usage_percent", "app_memory_usage_mb", 
            "disk_read_mb_per_sec", "disk_write_mb_per_sec",
            "gpu_memory_usage_percent"
        ]
        
        # Crear subplot con múltiples métricas
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[m.replace("_", " ").title() for m in performance_metrics[:6]],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
        
        for i, metric in enumerate(performance_metrics[:6]):
            if metric in metrics_data and metrics_data[metric]["values"]:
                row, col = positions[i]
                fig.add_trace(
                    go.Scatter(
                        x=metrics_data[metric]["timestamps"],
                        y=metrics_data[metric]["values"],
                        mode="lines+markers",
                        name=metric,
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return html.Div([
            html.H2("Análisis de Rendimiento"),
            dcc.Graph(figure=fig)
        ])
    
    def _create_alerts_tab(self, alerts_data):
        """Crea la pestaña de alertas."""
        
        if not alerts_data:
            return html.Div([
                html.H2("Alertas del Sistema"),
                html.Div("No hay alertas activas", className="no-alerts")
            ])
        
        # Crear tabla de alertas
        alert_rows = []
        for alert in alerts_data:
            severity_color = {
                "critical": "red",
                "high": "orange", 
                "medium": "yellow",
                "low": "lightblue"
            }.get(alert.get("severity", "low"), "gray")
            
            row = html.Tr([
                html.Td(alert.get("timestamp", ""), className="alert-timestamp"),
                html.Td(
                    html.Span(alert.get("severity", "").upper(), 
                             style={"color": severity_color, "font-weight": "bold"}),
                    className="alert-severity"
                ),
                html.Td(alert.get("title", ""), className="alert-title"),
                html.Td(alert.get("message", ""), className="alert-message"),
                html.Td(alert.get("component", ""), className="alert-component")
            ])
            alert_rows.append(row)
        
        alert_table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Timestamp"),
                    html.Th("Severidad"),
                    html.Th("Título"),
                    html.Th("Mensaje"),
                    html.Th("Componente")
                ])
            ]),
            html.Tbody(alert_rows)
        ], className="alerts-table")
        
        return html.Div([
            html.H2("Alertas del Sistema"),
            alert_table
        ])
    
    def _create_analysis_tab(self, metrics_data):
        """Crea la pestaña de análisis."""
        
        # Análisis de correlaciones
        correlation_data = self._calculate_correlations(metrics_data)
        
        # Gráfico de correlaciones
        correlation_fig = go.Figure(data=go.Heatmap(
            z=correlation_data["values"],
            x=correlation_data["metrics"],
            y=correlation_data["metrics"],
            colorscale="RdBu",
            zmid=0
        ))
        
        correlation_fig.update_layout(
            title="Matriz de Correlación de Métricas",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return html.Div([
            html.H2("Análisis Avanzado"),
            
            html.Div([
                html.H3("Correlación entre Métricas"),
                dcc.Graph(figure=correlation_fig)
            ]),
            
            html.Div([
                html.H3("Estadísticas del Sistema"),
                html.Div(id="system-statistics")
            ])
        ])
    
    def _create_time_series_chart(self, timestamps, values, title):
        """Crea un gráfico de serie temporal."""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode="lines+markers",
            name=title,
            line=dict(width=2, color="blue"),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        
        return fig
    
    def _create_trends_chart(self, metrics_data, metrics):
        """Crea un gráfico de tendencias múltiples."""
        
        fig = go.Figure()
        
        colors = ["blue", "red", "green", "orange", "purple", "brown"]
        
        for i, metric in enumerate(metrics):
            if metric in metrics_data and metrics_data[metric]["values"]:
                fig.add_trace(go.Scatter(
                    x=metrics_data[metric]["timestamps"],
                    y=metrics_data[metric]["values"],
                    mode="lines",
                    name=metric.replace("_", " ").title(),
                    line=dict(width=2, color=colors[i % len(colors)])
                ))
        
        fig.update_layout(
            title="Tendencias del Sistema",
            xaxis_title="Tiempo",
            yaxis_title="Valor (%)",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig
    
    def _calculate_correlations(self, metrics_data):
        """Calcula correlaciones entre métricas."""
        
        # Seleccionar métricas numéricas con suficientes datos
        numeric_metrics = []
        for metric_name, data in metrics_data.items():
            if data["values"] and len(data["values"]) > 10:
                try:
                    # Verificar que todos los valores son numéricos
                    float(data["values"][0])
                    numeric_metrics.append(metric_name)
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_metrics) < 2:
            return {"metrics": [], "values": []}
        
        # Crear DataFrame para cálculo de correlaciones
        try:
            import pandas as pd
            
            # Preparar datos
            min_length = min(len(metrics_data[m]["values"]) for m in numeric_metrics)
            data_dict = {}
            
            for metric in numeric_metrics:
                data_dict[metric] = metrics_data[metric]["values"][-min_length:]
            
            df = pd.DataFrame(data_dict)
            correlation_matrix = df.corr()
            
            return {
                "metrics": numeric_metrics,
                "values": correlation_matrix.values.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando correlaciones: {e}")
            return {"metrics": [], "values": []}
    
    def start_server(self):
        """Inicia el servidor web del dashboard."""
        
        if not self.app:
            self.logger.error("Aplicación Dash no inicializada")
            return
        
        self.running = True
        
        def run_server():
            try:
                self.app.run_server(
                    host=self.config.web_host,
                    port=self.config.web_port,
                    debug=False,
                    use_reloader=False
                )
            except Exception as e:
                self.logger.error(f"Error ejecutando servidor web: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Abrir navegador automáticamente
        time.sleep(2)  # Esperar a que inicie el servidor
        try:
            webbrowser.open(f"http://{self.config.web_host}:{self.config.web_port}")
        except Exception as e:
            self.logger.warning(f"No se pudo abrir el navegador automáticamente: {e}")
        
        self.logger.info(f"Dashboard web iniciado en http://{self.config.web_host}:{self.config.web_port}")
    
    def stop_server(self):
        """Detiene el servidor web del dashboard."""
        self.running = False
        self.logger.info("Dashboard web detenido")

class NativeDashboard(QMainWindow):
    """Dashboard nativo usando PyQt5."""
    
    def __init__(self, config: DashboardConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NativeDashboard")
        
        if not PYQT5_AVAILABLE:
            self.logger.error("PyQt5 no está disponible")
            return
        
        if not MONITORING_AVAILABLE:
            self.logger.error("Sistema de monitoreo no disponible")
            return
        
        self.monitoring_system = get_enhanced_monitoring_system()
        
        # Datos para gráficos
        self.metric_data = {}
        self.chart_series = {}
        
        # Timer para actualización
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_dashboard)
        
        self.setup_ui()
        self.setup_styling()
    
    def setup_ui(self):
        """Configura la interfaz de usuario."""
        
        self.setWindowTitle("SIGeC-Balistica - Dashboard de Monitoreo")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("SIGeC-Balistica - Dashboard de Monitoreo")
        title_label.setObjectName("title")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("Estado: Iniciando...")
        self.status_label.setObjectName("status")
        header_layout.addWidget(self.status_label)
        
        self.update_label = QLabel("Última actualización: --")
        self.update_label.setObjectName("update")
        header_layout.addWidget(self.update_label)
        
        main_layout.addLayout(header_layout)
        
        # Controles
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Intervalo (ms):"))
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1000, 30000)
        self.interval_spinbox.setValue(self.config.update_interval_ms)
        self.interval_spinbox.setSingleStep(1000)
        controls_layout.addWidget(self.interval_spinbox)
        
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setChecked(self.config.auto_refresh)
        controls_layout.addWidget(self.auto_refresh_checkbox)
        
        start_button = QPushButton("Iniciar Monitoreo")
        start_button.clicked.connect(self.start_monitoring)
        controls_layout.addWidget(start_button)
        
        stop_button = QPushButton("Detener Monitoreo")
        stop_button.clicked.connect(self.stop_monitoring)
        controls_layout.addWidget(stop_button)
        
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
        
        # Tabs
        self.tab_widget = QTabWidget()
        
        # Tab de vista general
        self.overview_tab = self.create_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "Vista General")
        
        # Tab de métricas
        self.metrics_tab = self.create_metrics_tab()
        self.tab_widget.addTab(self.metrics_tab, "Métricas")
        
        # Tab de alertas
        self.alerts_tab = self.create_alerts_tab()
        self.tab_widget.addTab(self.alerts_tab, "Alertas")
        
        main_layout.addWidget(self.tab_widget)
    
    def create_overview_tab(self):
        """Crea la pestaña de vista general."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Métricas clave con barras de progreso
        metrics_group = QGroupBox("Métricas Clave del Sistema")
        metrics_layout = QGridLayout(metrics_group)
        
        self.progress_bars = {}
        key_metrics = [
            ("CPU Usage", "cpu_usage_percent"),
            ("Memory Usage", "memory_usage_percent"),
            ("Disk Usage", "disk_usage_percent"),
            ("GPU Memory", "gpu_memory_usage_percent")
        ]
        
        for i, (label, metric_name) in enumerate(key_metrics):
            row = i // 2
            col = (i % 2) * 2
            
            metrics_layout.addWidget(QLabel(label), row, col)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            self.progress_bars[metric_name] = progress_bar
            metrics_layout.addWidget(progress_bar, row, col + 1)
        
        layout.addWidget(metrics_group)
        
        # Resumen de alertas
        alerts_group = QGroupBox("Resumen de Alertas")
        alerts_layout = QHBoxLayout(alerts_group)
        
        self.alert_labels = {}
        alert_types = ["Críticas", "Altas", "Medias", "Bajas"]
        
        for alert_type in alert_types:
            label = QLabel(f"{alert_type}: 0")
            label.setObjectName(f"alert_{alert_type.lower()}")
            self.alert_labels[alert_type] = label
            alerts_layout.addWidget(label)
        
        layout.addWidget(alerts_group)
        
        # Información del sistema
        system_group = QGroupBox("Información del Sistema")
        system_layout = QVBoxLayout(system_group)
        
        self.system_info_text = QTextEdit()
        self.system_info_text.setMaximumHeight(150)
        self.system_info_text.setReadOnly(True)
        system_layout.addWidget(self.system_info_text)
        
        layout.addWidget(system_group)
        
        return widget
    
    def create_metrics_tab(self):
        """Crea la pestaña de métricas."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Scroll area para múltiples gráficos
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Crear gráficos para métricas principales
        self.metric_charts = {}
        main_metrics = [
            "cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
            "load_average_1m", "process_count", "app_memory_usage_mb"
        ]
        
        for metric_name in main_metrics:
            chart_widget = self.create_metric_chart(metric_name)
            scroll_layout.addWidget(chart_widget)
            self.metric_charts[metric_name] = chart_widget
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_alerts_tab(self):
        """Crea la pestaña de alertas."""
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tabla de alertas
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(5)
        self.alerts_table.setHorizontalHeaderLabels([
            "Timestamp", "Severidad", "Título", "Mensaje", "Componente"
        ])
        
        # Configurar tabla
        header = self.alerts_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.alerts_table)
        
        return widget
    
    def create_metric_chart(self, metric_name):
        """Crea un gráfico para una métrica específica."""
        
        # Crear grupo para el gráfico
        group = QGroupBox(metric_name.replace("_", " ").title())
        layout = QVBoxLayout(group)
        
        # Por simplicidad, usar etiquetas en lugar de gráficos complejos
        # En una implementación completa, se usarían QChart y QChartView
        
        self.metric_labels = getattr(self, 'metric_labels', {})
        
        current_label = QLabel("Valor actual: --")
        current_label.setObjectName(f"current_{metric_name}")
        layout.addWidget(current_label)
        
        trend_label = QLabel("Tendencia: --")
        trend_label.setObjectName(f"trend_{metric_name}")
        layout.addWidget(trend_label)
        
        self.metric_labels[f"current_{metric_name}"] = current_label
        self.metric_labels[f"trend_{metric_name}"] = trend_label
        
        return group
    
    def setup_styling(self):
        """Configura el estilo del dashboard."""
        
        if self.config.theme == "dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                
                QLabel#title {
                    font-size: 18px;
                    font-weight: bold;
                    color: #4CAF50;
                }
                
                QLabel#status {
                    font-weight: bold;
                }
                
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 5px;
                    margin-top: 1ex;
                    padding-top: 10px;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                
                QProgressBar {
                    border: 2px solid #555555;
                    border-radius: 5px;
                    text-align: center;
                }
                
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
                
                QTableWidget {
                    gridline-color: #555555;
                    background-color: #3b3b3b;
                }
                
                QHeaderView::section {
                    background-color: #555555;
                    padding: 4px;
                    border: 1px solid #777777;
                }
            """)
    
    def start_monitoring(self):
        """Inicia el monitoreo."""
        
        try:
            self.monitoring_system.start_monitoring()
            
            # Configurar timer
            interval = self.interval_spinbox.value()
            self.update_timer.start(interval)
            
            self.status_label.setText("Estado: Monitoreando")
            self.logger.info("Dashboard nativo iniciado")
            
        except Exception as e:
            self.logger.error(f"Error iniciando monitoreo: {e}")
            self.status_label.setText("Estado: Error")
    
    def stop_monitoring(self):
        """Detiene el monitoreo."""
        
        self.update_timer.stop()
        self.monitoring_system.stop_monitoring()
        
        self.status_label.setText("Estado: Detenido")
        self.logger.info("Dashboard nativo detenido")
    
    def update_dashboard(self):
        """Actualiza el dashboard con nuevos datos."""
        
        if not self.auto_refresh_checkbox.isChecked():
            return
        
        try:
            # Obtener datos del sistema de monitoreo
            dashboard_data = self.monitoring_system.get_dashboard_data()
            
            # Actualizar estado del sistema
            system_status = dashboard_data.get("system_status", "unknown")
            self.status_label.setText(f"Estado: {system_status.upper()}")
            
            # Actualizar timestamp
            self.update_label.setText(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")
            
            # Actualizar métricas clave
            current_metrics = dashboard_data.get("current_metrics", {})
            
            for metric_name, progress_bar in self.progress_bars.items():
                if metric_name in current_metrics:
                    value = current_metrics[metric_name]["value"]
                    if isinstance(value, (int, float)):
                        progress_bar.setValue(int(value))
                        
                        # Cambiar color según el valor
                        if value > 90:
                            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }")
                        elif value > 80:
                            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ff9800; }")
                        else:
                            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
            
            # Actualizar etiquetas de métricas
            if hasattr(self, 'metric_labels'):
                for metric_name, metric_info in current_metrics.items():
                    current_key = f"current_{metric_name}"
                    trend_key = f"trend_{metric_name}"
                    
                    if current_key in self.metric_labels:
                        value = metric_info["value"]
                        unit = metric_info.get("unit", "")
                        self.metric_labels[current_key].setText(f"Valor actual: {value} {unit}")
                    
                    if trend_key in self.metric_labels:
                        trend = metric_info.get("trend", {}).get("trend", "unknown")
                        self.metric_labels[trend_key].setText(f"Tendencia: {trend}")
            
            # Actualizar alertas
            active_alerts = dashboard_data.get("active_alerts", [])
            self.update_alerts_table(active_alerts)
            
            # Actualizar contadores de alertas
            alert_counts = {"Críticas": 0, "Altas": 0, "Medias": 0, "Bajas": 0}
            
            for alert in active_alerts:
                severity = alert.get("severity", "low")
                if severity == "critical":
                    alert_counts["Críticas"] += 1
                elif severity == "high":
                    alert_counts["Altas"] += 1
                elif severity == "medium":
                    alert_counts["Medias"] += 1
                else:
                    alert_counts["Bajas"] += 1
            
            for alert_type, count in alert_counts.items():
                if alert_type in self.alert_labels:
                    self.alert_labels[alert_type].setText(f"{alert_type}: {count}")
            
            # Actualizar información del sistema
            system_info = self.monitoring_system.get_system_info()
            monitoring_stats = self.monitoring_system.get_monitoring_statistics()
            
            info_text = f"""
Tiempo de actividad: {system_info.get('uptime_hours', 0):.1f} horas
Recopilaciones: {system_info.get('collection_count', 0)}
Métricas únicas: {monitoring_stats.get('unique_metrics', 0)}
Memoria usada: {monitoring_stats.get('memory_usage_mb', 0):.1f} MB
Tasa de recopilación: {monitoring_stats.get('collection_rate_per_minute', 0):.1f} /min
            """.strip()
            
            self.system_info_text.setPlainText(info_text)
            
        except Exception as e:
            self.logger.error(f"Error actualizando dashboard: {e}")
    
    def update_alerts_table(self, alerts):
        """Actualiza la tabla de alertas."""
        
        self.alerts_table.setRowCount(len(alerts))
        
        for i, alert in enumerate(alerts):
            self.alerts_table.setItem(i, 0, QTableWidgetItem(alert.get("timestamp", "")))
            
            severity_item = QTableWidgetItem(alert.get("severity", "").upper())
            
            # Colorear según severidad
            severity = alert.get("severity", "low")
            if severity == "critical":
                severity_item.setBackground(QColor(244, 67, 54))  # Rojo
            elif severity == "high":
                severity_item.setBackground(QColor(255, 152, 0))  # Naranja
            elif severity == "medium":
                severity_item.setBackground(QColor(255, 235, 59))  # Amarillo
            
            self.alerts_table.setItem(i, 1, severity_item)
            self.alerts_table.setItem(i, 2, QTableWidgetItem(alert.get("title", "")))
            self.alerts_table.setItem(i, 3, QTableWidgetItem(alert.get("message", "")))
            self.alerts_table.setItem(i, 4, QTableWidgetItem(alert.get("component", "")))

class DashboardManager:
    """Gestor principal de dashboards."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        self.web_dashboard = None
        self.native_dashboard = None
        self.qt_app = None
        
    def start_web_dashboard(self):
        """Inicia el dashboard web."""
        
        if not DASH_AVAILABLE:
            self.logger.error("Dashboard web no disponible - Dash no instalado")
            return False
        
        if not self.config.enable_web_dashboard:
            self.logger.info("Dashboard web deshabilitado en configuración")
            return False
        
        try:
            self.web_dashboard = WebDashboard(self.config)
            self.web_dashboard.start_server()
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando dashboard web: {e}")
            return False
    
    def start_native_dashboard(self):
        """Inicia el dashboard nativo."""
        
        if not PYQT5_AVAILABLE:
            self.logger.error("Dashboard nativo no disponible - PyQt5 no instalado")
            return False
        
        if not self.config.enable_native_dashboard:
            self.logger.info("Dashboard nativo deshabilitado en configuración")
            return False
        
        try:
            # Crear aplicación Qt si no existe
            if not QApplication.instance():
                self.qt_app = QApplication([])
            
            self.native_dashboard = NativeDashboard(self.config)
            self.native_dashboard.show()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando dashboard nativo: {e}")
            return False
    
    def start_all_dashboards(self):
        """Inicia todos los dashboards disponibles."""
        
        web_started = self.start_web_dashboard()
        native_started = self.start_native_dashboard()
        
        if web_started:
            self.logger.info("Dashboard web iniciado correctamente")
        
        if native_started:
            self.logger.info("Dashboard nativo iniciado correctamente")
        
        if not web_started and not native_started:
            self.logger.error("No se pudo iniciar ningún dashboard")
            return False
        
        return True
    
    def run_native_dashboard(self):
        """Ejecuta el bucle principal del dashboard nativo."""
        
        if self.qt_app and self.native_dashboard:
            self.qt_app.exec_()
    
    def stop_all_dashboards(self):
        """Detiene todos los dashboards."""
        
        if self.web_dashboard:
            self.web_dashboard.stop_server()
        
        if self.native_dashboard:
            self.native_dashboard.stop_monitoring()
        
        self.logger.info("Todos los dashboards detenidos")

# Funciones de conveniencia
def create_dashboard_config(**kwargs) -> DashboardConfig:
    """Crea una configuración de dashboard personalizada."""
    config = DashboardConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

def start_dashboard(config: Optional[DashboardConfig] = None, web_only: bool = False, native_only: bool = False):
    """Inicia el dashboard con la configuración especificada."""
    
    manager = DashboardManager(config)
    
    if web_only:
        return manager.start_web_dashboard()
    elif native_only:
        success = manager.start_native_dashboard()
        if success:
            manager.run_native_dashboard()
        return success
    else:
        success = manager.start_all_dashboards()
        if manager.native_dashboard:
            manager.run_native_dashboard()
        return success

# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear configuración personalizada
    config = create_dashboard_config(
        web_port=8051,
        update_interval_ms=3000,
        theme="dark",
        enable_web_dashboard=True,
        enable_native_dashboard=True
    )
    
    # Iniciar dashboard
    start_dashboard(config)