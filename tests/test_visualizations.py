#!/usr/bin/env python3
"""
Script de prueba para verificar las visualizaciones integradas
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget
from PyQt5.QtCore import QTimer

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from gui.visualization_widgets import (
    PreprocessingVisualizationWidget,
    FeatureVisualizationWidget,
    StatisticalVisualizationWidget,
    ROIVisualizationWidget
)

class VisualizationTestWindow(QMainWindow):
    """Ventana de prueba para las visualizaciones"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prueba de Visualizaciones Integradas - SIGeC-Balística")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QVBoxLayout(central_widget)
        
        # Crear pestañas para diferentes visualizadores
        self.tab_widget = QTabWidget()
        
        # Pestaña de preprocesamiento
        self.preprocessing_widget = PreprocessingVisualizationWidget()
        self.tab_widget.addTab(self.preprocessing_widget, "🔧 Preprocesamiento")
        
        # Pestaña de características
        self.feature_widget = FeatureVisualizationWidget()
        self.tab_widget.addTab(self.feature_widget, "🎯 Características")
        
        # Pestaña estadística
        self.statistical_widget = StatisticalVisualizationWidget()
        self.tab_widget.addTab(self.statistical_widget, "📊 Estadísticas")
        
        # Pestaña ROI
        self.roi_widget = ROIVisualizationWidget()
        self.tab_widget.addTab(self.roi_widget, "🔍 ROI")
        
        layout.addWidget(self.tab_widget)
        
        # Conectar señales
        self.connect_signals()
        
        # Crear imagen de prueba después de un breve delay
        QTimer.singleShot(1000, self.create_test_data)
    
    def connect_signals(self):
        """Conecta las señales de los widgets"""
        # Preprocesamiento
        self.preprocessing_widget.visualizationReady.connect(
            lambda path: print(f"✅ Preprocesamiento listo: {path}")
        )
        self.preprocessing_widget.visualizationError.connect(
            lambda error: print(f"❌ Error en preprocesamiento: {error}")
        )
        
        # Características
        self.feature_widget.visualizationReady.connect(
            lambda path: print(f"✅ Características listas: {path}")
        )
        self.feature_widget.visualizationError.connect(
            lambda error: print(f"❌ Error en características: {error}")
        )
        
        # Estadísticas
        self.statistical_widget.visualizationReady.connect(
            lambda path: print(f"✅ Estadísticas listas: {path}")
        )
        self.statistical_widget.visualizationError.connect(
            lambda error: print(f"❌ Error en estadísticas: {error}")
        )
        
        # ROI
        self.roi_widget.visualizationReady.connect(
            lambda path: print(f"✅ ROI listo: {path}")
        )
        self.roi_widget.visualizationError.connect(
            lambda error: print(f"❌ Error en ROI: {error}")
        )
    
    def create_test_data(self):
        """Crea datos de prueba para las visualizaciones"""
        print("🔧 Creando datos de prueba...")
        
        # Crear imagen sintética de prueba
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Agregar algunos patrones para hacer más interesante
        cv2.circle(test_image, (200, 200), 50, (255, 255, 255), -1)
        cv2.rectangle(test_image, (100, 100), (300, 300), (128, 128, 128), 2)
        
        # Guardar imagen de prueba
        test_image_path = "/tmp/test_ballistic_image.png"
        cv2.imwrite(test_image_path, test_image)
        
        print(f"📸 Imagen de prueba creada: {test_image_path}")
        
        # Probar visualizaciones
        self.test_preprocessing(test_image_path)
        self.test_features(test_image_path)
        self.test_statistics()
        self.test_roi(test_image_path)
    
    def test_preprocessing(self, image_path: str):
        """Prueba la visualización de preprocesamiento"""
        print("🔧 Probando visualización de preprocesamiento...")
        
        # Usar el método correcto del widget
        self.preprocessing_widget.visualize_preprocessing(image_path, "cartridge_case")
    
    def test_features(self, image_path: str):
        """Prueba la visualización de características"""
        print("🎯 Probando visualización de características...")
        
        # Crear imagen de prueba y simular keypoints
        import cv2
        image = cv2.imread(image_path)
        
        # Simular keypoints como objetos cv2.KeyPoint (formato correcto)
        keypoints = [
            cv2.KeyPoint(x=100, y=100, size=10),
            cv2.KeyPoint(x=200, y=150, size=15),
            cv2.KeyPoint(x=300, y=200, size=12),
            cv2.KeyPoint(x=150, y=250, size=8),
            cv2.KeyPoint(x=250, y=300, size=20)
        ]
        
        self.feature_widget.visualize_keypoints(image, keypoints, "SIFT")
    
    def test_statistics(self):
        """Prueba la visualización estadística"""
        print("📊 Probando visualización estadística...")
        
        # Simular datos estadísticos en el formato esperado por StatisticalVisualizer
        analysis_results = {
            'correlation': {
                'correlation_matrix': np.array([
                    [1.0, 0.8, 0.6, 0.4],
                    [0.8, 1.0, 0.7, 0.5],
                    [0.6, 0.7, 1.0, 0.3],
                    [0.4, 0.5, 0.3, 1.0]
                ]),
                'feature_names': ['Característica 1', 'Característica 2', 'Característica 3', 'Característica 4'],
                'significant_correlations': [(0, 1, 0.8), (1, 2, 0.7)]
            },
            'clustering': {
                'labels': np.array([0, 0, 1, 1, 2, 2, 0, 1]),
                'n_clusters': 3,
                'silhouette_score': 0.75,
                'cluster_centers': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            },
            'pca': {
                'explained_variance_ratio': np.array([0.4, 0.3, 0.2, 0.1]),
                'components': np.array([
                    [0.5, 0.3, 0.2, 0.1],
                    [0.3, 0.5, 0.1, 0.2],
                    [0.2, 0.1, 0.5, 0.3],
                    [0.1, 0.2, 0.3, 0.5]
                ]),
                'transformed_data': np.random.randn(20, 4)
            }
        }
        
        # Datos de características originales
        features_data = [
            {'feature_1': 1.2, 'feature_2': 2.3, 'feature_3': 3.4, 'feature_4': 4.5},
            {'feature_1': 1.5, 'feature_2': 2.1, 'feature_3': 3.8, 'feature_4': 4.2},
            {'feature_1': 2.1, 'feature_2': 3.2, 'feature_3': 2.9, 'feature_4': 5.1},
            {'feature_1': 1.8, 'feature_2': 2.7, 'feature_3': 3.6, 'feature_4': 4.8}
        ]
        
        self.statistical_widget.create_interactive_dashboard(analysis_results)
    
    def test_roi(self, image_path: str):
        """Prueba la visualización de ROI"""
        print("🔍 Probando visualización de ROI...")
        
        # Simular regiones de interés con formato correcto
        roi_regions = [
            {
                'bbox': [150, 150, 100, 100],  # x, y, width, height
                'confidence': 0.95,
                'detection_method': 'cartridge_case',
                'area': 10000,
                'center': [200, 200]
            },
            {
                'bbox': [200, 250, 80, 80],
                'confidence': 0.87,
                'detection_method': 'enhanced_watershed',
                'area': 6400,
                'center': [240, 290]
            }
        ]
        
        self.roi_widget.visualize_roi_regions(image_path, roi_regions, "cartridge_case")

def main():
    """Función principal"""
    print("🚀 Iniciando prueba de visualizaciones...")
    
    app = QApplication(sys.argv)
    
    # Crear ventana de prueba
    window = VisualizationTestWindow()
    window.show()
    
    print("✅ Ventana de prueba creada")
    print("📋 Instrucciones:")
    print("   - Cambia entre las pestañas para ver diferentes visualizaciones")
    print("   - Observa la consola para mensajes de estado")
    print("   - Cierra la ventana para terminar la prueba")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()