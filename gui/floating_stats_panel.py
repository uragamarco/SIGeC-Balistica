"""
Panel flotante para estadísticas de matching balístico.
Convierte el panel de estadísticas en un panel flotante independiente.
Integrado con AppStateManager para sincronización de estado.
"""

from PyQt5.QtWidgets import (QGroupBox, QGridLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QLabel,
                             QVBoxLayout, QHBoxLayout, QFrame, QPushButton,
                             QDockWidget, QWidget, QSplitter, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

# Importar AppStateManager si está disponible
try:
    from .app_state_manager import AppStateManager
    APP_STATE_AVAILABLE = True
except ImportError:
    APP_STATE_AVAILABLE = False


class FloatingStatsPanel(QDockWidget):
    """
    Panel flotante para estadísticas de matching balístico.
    Muestra métricas detalladas del proceso de comparación.
    Compatible con QDockWidget para mejor integración.
    """
    
    # Señales para comunicación
    stats_updated = pyqtSignal(dict)
    keypoint_selected = pyqtSignal(dict)
    panel_closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("📊 Estadísticas del Match", parent)
        
        # Variables para información de keypoints
        self.current_keypoint_info = None
        self.current_stats = {}
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._show_keypoint_details)
        
        # Configurar el dock widget
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        
        # Crear la interfaz completa
        self._create_stats_interface()
        
        # Configurar tamaño inicial
        self.resize(420, 380)
        
        # Conectar con AppStateManager si está disponible
        if APP_STATE_AVAILABLE:
            self.app_state = AppStateManager()
            # Usar las señales que realmente existen en AppStateManager
            self.app_state.statistics_updated.connect(self.update_statistics)
        
    def _create_stats_interface(self):
        """Crear la interfaz completa de estadísticas."""
        # Widget principal
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # === SECCIÓN 1: RESUMEN RÁPIDO ===
        self._create_quick_summary()
        main_layout.addWidget(self.summary_frame)
        
        # === SECCIÓN 2: ESTADÍSTICAS DETALLADAS ===
        self._create_stats_table()
        main_layout.addWidget(self.stats_table)
        
        # === SECCIÓN 3: INFORMACIÓN DE KEYPOINT (HOVER) ===
        self._create_keypoint_info_section()
        main_layout.addWidget(self.keypoint_info_frame)
        
        # === SECCIÓN 4: CONTROLES ADICIONALES ===
        self._create_controls_section()
        main_layout.addWidget(self.controls_frame)
        
        # Establecer el widget principal
        self.setWidget(main_widget)
        
    def _create_quick_summary(self):
        """Crear el resumen rápido."""
        self.summary_frame = QFrame()
        self.summary_frame.setFrameStyle(QFrame.StyledPanel)
        self.summary_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4A90E2, stop: 1 #357ABD);
                border-radius: 8px;
                padding: 8px;
                margin: 2px;
            }
        """)
        
        layout = QHBoxLayout(self.summary_frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(15)
        
        # Etiquetas de resumen
        self.similarity_summary = QLabel("Similitud: --")
        self.similarity_summary.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        
        self.matches_summary = QLabel("Matches: --")
        self.matches_summary.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        
        self.quality_summary = QLabel("Calidad: --")
        self.quality_summary.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        
        # Separadores
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: rgba(255,255,255,0.7); font-weight: bold;")
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: rgba(255,255,255,0.7); font-weight: bold;")
        
        layout.addWidget(self.similarity_summary)
        layout.addWidget(sep1)
        layout.addWidget(self.matches_summary)
        layout.addWidget(sep2)
        layout.addWidget(self.quality_summary)
        layout.addStretch()
        
    def _create_stats_table(self):
        """Crear la tabla de estadísticas."""
        # Crear tabla de estadísticas con mejor estilo
        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["📊 Métrica", "📈 Valor"])
        
        # Configurar tabla
        header = self.stats_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultSectionSize(140)
        
        self.stats_table.setMaximumHeight(200)
        self.stats_table.setMinimumHeight(120)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Aplicar estilos mejorados
        self.stats_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                gridline-color: #E8E8E8;
                font-size: 11px;
                selection-background-color: #4A90E2;
            }
            QTableWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #E8E8E8;
            }
            QTableWidget::item:selected {
                background-color: #4A90E2;
                color: white;
            }
            QTableWidget::item:alternate {
                background-color: #F8F9FA;
            }
            QHeaderView::section {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4A90E2, stop: 1 #357ABD);
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
                font-size: 11px;
            }
            QHeaderView::section:first {
                border-top-left-radius: 6px;
            }
            QHeaderView::section:last {
                border-top-right-radius: 6px;
            }
        """)
        
    def _create_keypoint_info_section(self):
        """Crear la sección de información de keypoints."""
        self.keypoint_info_frame = QFrame()
        self.keypoint_info_frame.setFrameStyle(QFrame.StyledPanel)
        self.keypoint_info_frame.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                padding: 8px;
                margin: 2px;
            }
        """)
        self.keypoint_info_frame.setVisible(False)  # Oculto por defecto
        
        layout = QVBoxLayout(self.keypoint_info_frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Título de la sección
        title_label = QLabel("🎯 Información del Keypoint")
        title_label.setStyleSheet("font-weight: bold; color: #2C3E50; font-size: 12px;")
        layout.addWidget(title_label)
        
        # Información detallada
        self.keypoint_details_label = QLabel("Pase el cursor sobre un keypoint para ver detalles...")
        self.keypoint_details_label.setStyleSheet("""
            QLabel {
                color: #5D6D7E;
                font-size: 10px;
                padding: 6px;
                background-color: white;
                border-radius: 4px;
                border: 1px solid #E8E8E8;
            }
        """)
        self.keypoint_details_label.setWordWrap(True)
        layout.addWidget(self.keypoint_details_label)
        
    def _create_controls_section(self):
        """Crear la sección de controles adicionales."""
        self.controls_frame = QFrame()
        self.controls_frame.setFrameStyle(QFrame.StyledPanel)
        self.controls_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #D0D0D0;
                border-radius: 6px;
                padding: 6px;
                margin: 2px;
            }
        """)
        
        layout = QHBoxLayout(self.controls_frame)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        
        # Botón para exportar estadísticas
        self.export_btn = QPushButton("📤 Exportar")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357ABD;
            }
            QPushButton:pressed {
                background-color: #2E6DA4;
            }
        """)
        self.export_btn.clicked.connect(self._export_statistics)
        
        # Botón para limpiar estadísticas
        self.clear_btn = QPushButton("🗑️ Limpiar")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:pressed {
                background-color: #A93226;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_statistics)
        
        layout.addWidget(self.export_btn)
        layout.addWidget(self.clear_btn)
        layout.addStretch()
        
    def update_statistics(self, stats_data):
        """
        Actualizar las estadísticas mostradas en la tabla.
        
        Args:
            stats_data (dict): Diccionario con las métricas a mostrar
        """
        if not stats_data:
            self.clear_statistics()
            return
            
        self.current_stats = stats_data.copy()
        
        # Limpiar tabla existente
        self.stats_table.setRowCount(0)
        
        # Mapeo de estadísticas con iconos y formato mejorado
        stats_mapping = {
            'total_matches': ('🎯 Total de coincidencias', lambda x: f"{x:,}"),
            'valid_matches': ('✅ Coincidencias válidas', lambda x: f"{x:,}"),
            'average_quality': ('⭐ Calidad promedio', lambda x: f"{x:.1f}%"),
            'similarity_score': ('🔍 Puntuación de similitud', lambda x: f"{x:.2f}%"),
            'processing_time': ('⏱️ Tiempo de procesamiento', lambda x: f"{x:.3f}s"),
            'keypoints_img1': ('🎯 Puntos clave (Img 1)', lambda x: f"{x:,}"),
            'keypoints_img2': ('🎯 Puntos clave (Img 2)', lambda x: f"{x:,}"),
            'match_ratio': ('📊 Ratio de coincidencias', lambda x: f"{x:.2f}%"),
            'algorithm_used': ('🔧 Algoritmo utilizado', lambda x: str(x)),
            'nist_quality': ('📋 Calidad NIST', lambda x: f"{x:.1f}"),
            'afte_conclusion': ('⚖️ Conclusión AFTE', lambda x: str(x))
        }
        
        row = 0
        for metric, value in stats_data.items():
            if metric in stats_mapping:
                label, formatter = stats_mapping[metric]
                
                self.stats_table.insertRow(row)
                
                # Crear items con mejor formato
                metric_item = QTableWidgetItem(label)
                value_item = QTableWidgetItem(formatter(value))
                value_item.setTextAlignment(Qt.AlignCenter)
                
                self.stats_table.setItem(row, 0, metric_item)
                self.stats_table.setItem(row, 1, value_item)
                row += 1
        
        # Ajustar el tamaño de las columnas
        self.stats_table.resizeColumnsToContents()
        
        # Actualizar resumen rápido
        self._update_quick_summary(stats_data)
        
        # Emitir señal de actualización
        self.stats_updated.emit(stats_data)
        
        # Actualizar AppStateManager si está disponible
        if APP_STATE_AVAILABLE and hasattr(self, 'app_state'):
            self.app_state.update_comparison_stats(stats_data)
        
    def _update_quick_summary(self, stats_data):
        """Actualizar el resumen rápido."""
        # Similitud
        similarity = stats_data.get('similarity_score', stats_data.get('Similitud', 0))
        if isinstance(similarity, (int, float)):
            if similarity <= 1.0:  # Asumiendo que está en escala 0-1
                similarity_text = f"Similitud: {similarity:.1%}"
                similarity_val = similarity
            else:  # Asumiendo que está en escala 0-100
                similarity_text = f"Similitud: {similarity:.1f}%"
                similarity_val = similarity / 100.0
                
            # Colores basados en el valor de similitud
            if similarity_val > 0.8:
                similarity_color = "#2ECC71"  # Verde
            elif similarity_val > 0.5:
                similarity_color = "#F39C12"  # Naranja
            else:
                similarity_color = "#E74C3C"  # Rojo
        else:
            similarity_text = f"Similitud: {similarity}"
            similarity_color = "white"
            
        self.similarity_summary.setText(similarity_text)
        self.similarity_summary.setStyleSheet(f"color: {similarity_color}; font-weight: bold; font-size: 12px;")
        
        # Matches
        matches = stats_data.get('total_matches', stats_data.get('valid_matches', stats_data.get('Matches Totales', 0)))
        self.matches_summary.setText(f"Matches: {matches}")
        
        # Calidad
        quality = stats_data.get('average_quality', stats_data.get('nist_quality', stats_data.get('Calidad Promedio', 0)))
        if isinstance(quality, (int, float)):
            quality_text = f"Calidad: {quality:.1f}"
        else:
            quality_text = f"Calidad: {quality}"
        self.quality_summary.setText(quality_text)
        
    def show_keypoint_info(self, keypoint_data, delay_ms=500):
        """Mostrar información detallada de un keypoint con delay."""
        self.current_keypoint_info = keypoint_data
        self.hover_timer.start(delay_ms)
        
    def hide_keypoint_info(self):
        """Ocultar información de keypoint."""
        self.hover_timer.stop()
        self.keypoint_info_frame.setVisible(False)
        
    def _show_keypoint_details(self):
        """Mostrar los detalles del keypoint actual."""
        if not self.current_keypoint_info:
            return
            
        kp_data = self.current_keypoint_info
        
        # Formatear información del keypoint
        details = []
        
        if 'position' in kp_data:
            pos = kp_data['position']
            details.append(f"📍 Posición: ({pos[0]:.1f}, {pos[1]:.1f})")
            
        if 'response' in kp_data:
            details.append(f"💪 Respuesta: {kp_data['response']:.3f}")
            
        if 'angle' in kp_data:
            details.append(f"🔄 Ángulo: {kp_data['angle']:.1f}°")
            
        if 'scale' in kp_data:
            details.append(f"📏 Escala: {kp_data['scale']:.3f}")
            
        if 'match_distance' in kp_data:
            distance = kp_data['match_distance']
            quality = "Excelente" if distance < 0.3 else "Buena" if distance < 0.6 else "Regular"
            details.append(f"🎯 Distancia: {distance:.3f} ({quality})")
            
        if 'descriptor_similarity' in kp_data:
            sim = kp_data['descriptor_similarity']
            details.append(f"🔍 Similitud: {sim:.1%}")
            
        # Mostrar información
        if details:
            self.keypoint_details_label.setText("\n".join(details))
            self.keypoint_info_frame.setVisible(True)
            # Emitir señal de selección
            self.keypoint_selected.emit(kp_data)
            
    def clear_statistics(self):
        """Limpiar todas las estadísticas de la tabla."""
        self.stats_table.setRowCount(0)
        self.current_stats = {}
        
        # Resetear resumen rápido
        self.similarity_summary.setText("Similitud: --")
        self.matches_summary.setText("Matches: --")
        self.quality_summary.setText("Calidad: --")
        
        # Ocultar información de keypoint
        self.hide_keypoint_info()
        
    def _export_statistics(self):
        """Exportar estadísticas actuales a archivo JSON."""
        if not self.current_stats:
            return
            
        from PyQt5.QtWidgets import QFileDialog
        import json
        from datetime import datetime
        
        # Seleccionar archivo de destino
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Estadísticas",
            f"estadisticas_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "Archivos JSON (*.json)"
        )
        
        if filename:
            try:
                # Preparar datos para exportación
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.current_stats,
                    'export_source': 'FloatingStatsPanel'
                }
                
                # Guardar archivo
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
                # Mostrar confirmación
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Exportación Exitosa",
                    f"Estadísticas exportadas correctamente a:\n{filename}"
                )
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Error de Exportación",
                    f"No se pudieron exportar las estadísticas:\n{str(e)}"
                )
                
    def get_statistics_data(self):
        """
        Obtener los datos actuales de estadísticas.
        
        Returns:
            dict: Diccionario con las estadísticas actuales
        """
        return self.current_stats.copy()
        
    def add_custom_statistic(self, metric, value):
        """
        Añadir una estadística personalizada.
        
        Args:
            metric (str): Nombre de la métrica
            value: Valor de la métrica
        """
        if not hasattr(self, 'current_stats'):
            self.current_stats = {}
            
        self.current_stats[metric] = value
        self.update_statistics(self.current_stats)
        
    def set_default_statistics(self):
        """Establecer estadísticas por defecto cuando no hay datos."""
        default_stats = {
            'similarity_score': 0.0,
            'total_matches': 0,
            'valid_matches': 0,
            'processing_time': 0.0,
            'algorithm_used': 'ORB',
            'keypoints_img1': 0,
            'keypoints_img2': 0,
            'match_ratio': 0.0
        }
        
        self.update_statistics(default_stats)
        
    def closeEvent(self, event):
        """Manejar el evento de cierre del panel."""
        self.panel_closed.emit()
        super().closeEvent(event)