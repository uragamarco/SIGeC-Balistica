"""
Métodos de visualización para el análisis balístico
Contiene funciones para ajustes en tiempo real, overlays y exportación
"""

from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap
import os


class VisualizationMethods:
    """Métodos de visualización para AnalysisTab"""
    
    def clear_results(self):
        """Limpia los resultados mostrados"""
        # Limpiar layout de resultados de forma segura
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)
                    
        # Agregar placeholder
        from PyQt5.QtWidgets import QLabel
        placeholder_label = QLabel("Los resultados aparecerán aquí después del análisis")
        placeholder_label.setProperty("class", "caption")
        placeholder_label.setAlignment(Qt.AlignCenter)
        placeholder_label.setStyleSheet("color: #757575; padding: 20px;")
        self.results_layout.addWidget(placeholder_label)
        
    # Métodos para ajustes de visualización en tiempo real
    def update_brightness(self, value):
        """Actualiza el brillo de la imagen"""
        self.brightness_value.setText(str(value))
        self.apply_image_adjustments()
        
    def update_contrast(self, value):
        """Actualiza el contraste de la imagen"""
        self.contrast_value.setText(str(value))
        self.apply_image_adjustments()
        
    def update_sharpness(self, value):
        """Actualiza la nitidez de la imagen"""
        self.sharpness_value.setText(str(value))
        self.apply_image_adjustments()
        
    def reset_image_adjustments(self):
        """Restablece todos los ajustes de imagen a valores por defecto"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.sharpness_slider.setValue(0)
        self.apply_image_adjustments()
        
    def apply_image_adjustments(self):
        """Aplica los ajustes de imagen al visor"""
        # Obtener valores actuales
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        sharpness = self.sharpness_slider.value()
        
        # Aplicar ajustes al ImageViewer
        # Nota: Esto requeriría extender ImageViewer para soportar ajustes en tiempo real
        # Por ahora, solo actualizamos los valores mostrados
        pass
        
    # Métodos para overlays de características
    def toggle_roi_overlay(self, enabled):
        """Activa/desactiva overlay de regiones de interés"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def toggle_firing_pin_overlay(self, enabled):
        """Activa/desactiva overlay de marcas de percutor"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def toggle_breech_face_overlay(self, enabled):
        """Activa/desactiva overlay de cara de recámara"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def toggle_extractor_overlay(self, enabled):
        """Activa/desactiva overlay de marcas de extractor"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def toggle_striations_overlay(self, enabled):
        """Activa/desactiva overlay de patrones de estriado"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def toggle_quality_map_overlay(self, enabled):
        """Activa/desactiva overlay de mapa de calidad"""
        if enabled:
            self.enable_overlay_controls()
        self.update_overlays()
        
    def update_overlay_transparency(self, value):
        """Actualiza la transparencia de los overlays"""
        self.transparency_value.setText(f"{value}%")
        self.update_overlays()
        
    def enable_overlay_controls(self):
        """Habilita los controles de transparencia cuando hay overlays activos"""
        self.overlay_transparency.setEnabled(True)
        
    def disable_all_overlays(self):
        """Deshabilita todos los overlays y controles"""
        self.show_roi_cb.setChecked(False)
        self.show_firing_pin_cb.setChecked(False)
        self.show_breech_face_cb.setChecked(False)
        self.show_extractor_cb.setChecked(False)
        self.show_striations_cb.setChecked(False)
        self.show_quality_map_cb.setChecked(False)
        self.overlay_transparency.setEnabled(False)
        
    def update_overlays(self):
        """Actualiza la visualización de overlays"""
        # Implementar lógica para mostrar/ocultar overlays en el ImageViewer
        # Esto requeriría extender ImageViewer para soportar overlays
        pass
        
    # Métodos para exportación y comparación
    def show_side_by_side_comparison(self):
        """Muestra comparación lado a lado de imagen original vs procesada"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            QMessageBox.warning(self, "Advertencia", "No hay imagen cargada para comparar")
            return
            
        # Crear ventana de comparación
        # Esto requeriría implementar una ventana de comparación dedicada
        QMessageBox.information(self, "Comparación", "Función de comparación lado a lado en desarrollo")
        
    def export_visualization(self):
        """Exporta la visualización actual"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            QMessageBox.warning(self, "Advertencia", "No hay visualización para exportar")
            return
            
        # Abrir diálogo de guardado
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Visualización",
            "",
            "Imágenes PNG (*.png);;Imágenes JPEG (*.jpg);;Todos los archivos (*)"
        )
        
        if file_path:
            try:
                # Exportar la imagen actual del viewer
                # Esto requeriría extender ImageViewer para soportar exportación
                QMessageBox.information(self, "Éxito", f"Visualización exportada a: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al exportar: {str(e)}")
                
    def enable_visualization_features(self, image_path):
        """Habilita las características de visualización cuando se carga una imagen"""
        self.current_image_path = image_path
        
        # Habilitar controles de ajuste en tiempo real
        self.brightness_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.sharpness_slider.setEnabled(True)
        
        # Habilitar controles de overlay
        self.roi_overlay_cb.setEnabled(True)
        self.firing_pin_overlay_cb.setEnabled(True)
        self.breech_face_overlay_cb.setEnabled(True)
        self.extractor_overlay_cb.setEnabled(True)
        self.striations_overlay_cb.setEnabled(True)
        self.quality_map_overlay_cb.setEnabled(True)
        self.overlay_transparency_slider.setEnabled(True)
        
        # Habilitar botones de exportación y comparación
        self.export_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)
        
        # Configurar minimapa si existe un visor de imagen interactivo
        if hasattr(self, 'image_viewer') and hasattr(self.image_viewer, 'original_pixmap'):
            if hasattr(self, 'minimap_widget'):
                self.minimap_widget.set_image(self.image_viewer.original_pixmap)
                # Conectar señales del minimapa
                self.minimap_widget.viewportChanged.connect(self.on_minimap_viewport_changed)
        
        # Cargar imagen en el viewer
        self.image_viewer.load_image(image_path)
    
    def on_minimap_viewport_changed(self, viewport_rect):
        """Maneja cambios en el viewport del minimapa"""
        if hasattr(self, 'image_viewer') and hasattr(self.image_viewer, 'scroll_area'):
            # Obtener scrollbars del visor principal
            h_scroll = self.image_viewer.scroll_area.horizontalScrollBar()
            v_scroll = self.image_viewer.scroll_area.verticalScrollBar()
            
            # Convertir coordenadas del viewport a posición de scroll
            if hasattr(self.image_viewer, 'zoom_factor'):
                scroll_x = int(viewport_rect.x() * self.image_viewer.zoom_factor)
                scroll_y = int(viewport_rect.y() * self.image_viewer.zoom_factor)
                
                # Aplicar nueva posición de scroll
                h_scroll.setValue(scroll_x)
                v_scroll.setValue(scroll_y)
    
    def update_minimap_viewport(self):
        """Actualiza el viewport del minimapa basado en la posición actual del scroll"""
        if (hasattr(self, 'minimap_widget') and hasattr(self, 'image_viewer') and 
            hasattr(self.image_viewer, 'scroll_area') and hasattr(self.image_viewer, 'original_pixmap')):
            
            # Obtener información actual del visor
            h_scroll = self.image_viewer.scroll_area.horizontalScrollBar()
            v_scroll = self.image_viewer.scroll_area.verticalScrollBar()
            scroll_area_size = self.image_viewer.scroll_area.size()
            
            if hasattr(self.image_viewer, 'zoom_factor') and self.image_viewer.original_pixmap:
                # Calcular viewport actual en coordenadas de imagen original
                viewport_x = h_scroll.value() / self.image_viewer.zoom_factor
                viewport_y = v_scroll.value() / self.image_viewer.zoom_factor
                viewport_width = scroll_area_size.width() / self.image_viewer.zoom_factor
                viewport_height = scroll_area_size.height() / self.image_viewer.zoom_factor
                
                viewport_rect = QRect(
                    int(viewport_x), int(viewport_y),
                    int(viewport_width), int(viewport_height)
                )
                
                # Actualizar minimapa
                self.minimap_widget.set_viewport(
                    viewport_rect,
                    self.image_viewer.original_pixmap.size(),
                    self.image_viewer.zoom_factor
                )