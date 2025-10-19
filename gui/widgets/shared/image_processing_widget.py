"""
ImageProcessingWidget - Shared image processing configuration component
Eliminates code duplication between analysis and comparison tabs
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QCheckBox, QRadioButton, QSlider,
                             QLabel, QFrame, QButtonGroup, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTextEdit, QPushButton,
                             QTabWidget, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor
from typing import Dict, Any


class ImageProcessingWidget(QWidget):
    """Shared image processing configuration widget"""
    
    configurationChanged = pyqtSignal(dict)  # Emitted when configuration changes
    previewRequested = pyqtSignal(dict)      # Emitted when preview is requested
    
    def __init__(self, mode: str = "analysis", parent=None):
        """
        Initialize image processing configuration widget
        
        Args:
            mode: "analysis" or "comparison" to customize labels and options
        """
        super().__init__(parent)
        self.mode = mode
        self.config = {}
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Main enable checkbox
        self.enable_processing_cb = QCheckBox(
            "Activar procesamiento de imágenes" + 
            (" para comparación" if self.mode == "comparison" else " para análisis")
        )
        self.enable_processing_cb.setProperty("class", "img-enable-checkbox")
        layout.addWidget(self.enable_processing_cb)
        
        # Main configuration panel
        self.main_panel = QFrame()
        self.main_panel.setProperty("class", "img-config-panel")
        self.main_panel.setEnabled(False)
        
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # Create tabbed interface for different processing categories
        self.tabs = QTabWidget()
        self.tabs.setProperty("class", "img-tabs")
        
        # Enhancement tab
        enhancement_tab = self.create_enhancement_tab()
        self.tabs.addTab(enhancement_tab, "Mejora")
        
        # Filtering tab
        filtering_tab = self.create_filtering_tab()
        self.tabs.addTab(filtering_tab, "Filtros")
        
        # Morphology tab
        morphology_tab = self.create_morphology_tab()
        self.tabs.addTab(morphology_tab, "Morfología")
        
        # Advanced tab
        advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(advanced_tab, "Avanzado")
        
        main_layout.addWidget(self.tabs)
        
        # Preview and control buttons
        control_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("Vista Previa")
        self.preview_btn.setProperty("class", "img-button primary")
        
        self.reset_btn = QPushButton("Restablecer")
        self.reset_btn.setProperty("class", "img-button secondary")
        
        self.apply_btn = QPushButton("Aplicar Configuración")
        self.apply_btn.setProperty("class", "img-button success")
        
        control_layout.addWidget(self.preview_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.apply_btn)
        
        main_layout.addLayout(control_layout)
        
        layout.addWidget(self.main_panel)
        
    def create_enhancement_tab(self) -> QWidget:
        """Create image enhancement tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Brightness and Contrast
        brightness_group = QGroupBox("Brillo y Contraste")
        brightness_group.setProperty("class", "img-section")
        brightness_layout = QVBoxLayout(brightness_group)
        
        # Brightness
        brightness_frame = QFrame()
        brightness_frame_layout = QVBoxLayout(brightness_frame)
        
        brightness_label = QLabel("Brillo")
        brightness_label.setProperty("class", "img-label")
        brightness_frame_layout.addWidget(brightness_label)
        
        brightness_slider_layout = QHBoxLayout()
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setProperty("class", "img-slider")
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        
        self.brightness_value_label = QLabel("0")
        self.brightness_value_label.setProperty("class", "img-value-label")
        self.brightness_value_label.setMinimumWidth(30)
        
        brightness_slider_layout.addWidget(self.brightness_slider)
        brightness_slider_layout.addWidget(self.brightness_value_label)
        brightness_frame_layout.addLayout(brightness_slider_layout)
        
        # Contrast
        contrast_frame = QFrame()
        contrast_frame_layout = QVBoxLayout(contrast_frame)
        
        contrast_label = QLabel("Contraste")
        contrast_label.setProperty("class", "img-label")
        contrast_frame_layout.addWidget(contrast_label)
        
        contrast_slider_layout = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setProperty("class", "img-slider")
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        
        self.contrast_value_label = QLabel("0")
        self.contrast_value_label.setProperty("class", "img-value-label")
        self.contrast_value_label.setMinimumWidth(30)
        
        contrast_slider_layout.addWidget(self.contrast_slider)
        contrast_slider_layout.addWidget(self.contrast_value_label)
        contrast_frame_layout.addLayout(contrast_slider_layout)
        
        brightness_layout.addWidget(brightness_frame)
        brightness_layout.addWidget(contrast_frame)
        layout.addWidget(brightness_group)
        
        # Gamma Correction
        gamma_group = QGroupBox("Corrección Gamma")
        gamma_group.setProperty("class", "img-section")
        gamma_layout = QVBoxLayout(gamma_group)
        
        self.gamma_enable_cb = QCheckBox("Activar corrección gamma")
        self.gamma_enable_cb.setProperty("class", "img-option")
        gamma_layout.addWidget(self.gamma_enable_cb)
        
        gamma_frame = QFrame()
        gamma_frame.setEnabled(False)
        gamma_frame_layout = QVBoxLayout(gamma_frame)
        
        gamma_slider_layout = QHBoxLayout()
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setProperty("class", "img-slider")
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        
        self.gamma_value_label = QLabel("1.0")
        self.gamma_value_label.setProperty("class", "img-value-label")
        self.gamma_value_label.setMinimumWidth(30)
        
        gamma_slider_layout.addWidget(self.gamma_slider)
        gamma_slider_layout.addWidget(self.gamma_value_label)
        gamma_frame_layout.addLayout(gamma_slider_layout)
        
        gamma_layout.addWidget(gamma_frame)
        self.gamma_frame = gamma_frame
        layout.addWidget(gamma_group)
        
        # Histogram Equalization
        histogram_group = QGroupBox("Ecualización de Histograma")
        histogram_group.setProperty("class", "img-section")
        histogram_layout = QVBoxLayout(histogram_group)
        
        self.histogram_eq_cb = QCheckBox("Ecualización global")
        self.histogram_eq_cb.setProperty("class", "img-option")
        histogram_layout.addWidget(self.histogram_eq_cb)
        
        self.clahe_cb = QCheckBox("CLAHE (Ecualización adaptativa)")
        self.clahe_cb.setProperty("class", "img-option")
        histogram_layout.addWidget(self.clahe_cb)
        
        # CLAHE parameters
        clahe_frame = QFrame()
        clahe_frame.setEnabled(False)
        clahe_layout = QFormLayout(clahe_frame)
        
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setProperty("class", "img-spin")
        self.clahe_clip_spin.setRange(1.0, 10.0)
        self.clahe_clip_spin.setSingleStep(0.5)
        self.clahe_clip_spin.setValue(2.0)
        clahe_layout.addRow("Límite de Clip:", self.clahe_clip_spin)
        
        self.clahe_grid_spin = QSpinBox()
        self.clahe_grid_spin.setProperty("class", "img-spin")
        self.clahe_grid_spin.setRange(2, 16)
        self.clahe_grid_spin.setValue(8)
        clahe_layout.addRow("Tamaño de Grilla:", self.clahe_grid_spin)
        
        histogram_layout.addWidget(clahe_frame)
        self.clahe_frame = clahe_frame
        layout.addWidget(histogram_group)
        
        layout.addStretch()
        return widget
        
    def create_filtering_tab(self) -> QWidget:
        """Create filtering tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Noise Reduction
        noise_group = QGroupBox("Reducción de Ruido")
        noise_group.setProperty("class", "img-section")
        noise_layout = QVBoxLayout(noise_group)
        
        # Gaussian blur
        self.gaussian_blur_cb = QCheckBox("Desenfoque Gaussiano")
        self.gaussian_blur_cb.setProperty("class", "img-option")
        noise_layout.addWidget(self.gaussian_blur_cb)
        
        gaussian_frame = QFrame()
        gaussian_frame.setEnabled(False)
        gaussian_layout = QFormLayout(gaussian_frame)
        
        self.gaussian_kernel_spin = QSpinBox()
        self.gaussian_kernel_spin.setProperty("class", "img-spin")
        self.gaussian_kernel_spin.setRange(3, 15)
        self.gaussian_kernel_spin.setSingleStep(2)
        self.gaussian_kernel_spin.setValue(5)
        gaussian_layout.addRow("Tamaño de Kernel:", self.gaussian_kernel_spin)
        
        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setProperty("class", "img-spin")
        self.gaussian_sigma_spin.setRange(0.1, 5.0)
        self.gaussian_sigma_spin.setSingleStep(0.1)
        self.gaussian_sigma_spin.setValue(1.0)
        gaussian_layout.addRow("Sigma:", self.gaussian_sigma_spin)
        
        noise_layout.addWidget(gaussian_frame)
        self.gaussian_frame = gaussian_frame
        
        # Median filter
        self.median_filter_cb = QCheckBox("Filtro Mediano")
        self.median_filter_cb.setProperty("class", "img-option")
        noise_layout.addWidget(self.median_filter_cb)
        
        median_frame = QFrame()
        median_frame.setEnabled(False)
        median_layout = QFormLayout(median_frame)
        
        self.median_kernel_spin = QSpinBox()
        self.median_kernel_spin.setProperty("class", "img-spin")
        self.median_kernel_spin.setRange(3, 15)
        self.median_kernel_spin.setSingleStep(2)
        self.median_kernel_spin.setValue(5)
        median_layout.addRow("Tamaño de Kernel:", self.median_kernel_spin)
        
        noise_layout.addWidget(median_frame)
        self.median_frame = median_frame
        
        # Bilateral filter
        self.bilateral_cb = QCheckBox("Filtro Bilateral")
        self.bilateral_cb.setProperty("class", "img-option")
        noise_layout.addWidget(self.bilateral_cb)
        
        bilateral_frame = QFrame()
        bilateral_frame.setEnabled(False)
        bilateral_layout = QFormLayout(bilateral_frame)
        
        self.bilateral_d_spin = QSpinBox()
        self.bilateral_d_spin.setProperty("class", "img-spin")
        self.bilateral_d_spin.setRange(5, 25)
        self.bilateral_d_spin.setValue(9)
        bilateral_layout.addRow("Diámetro:", self.bilateral_d_spin)
        
        self.bilateral_sigma_color_spin = QDoubleSpinBox()
        self.bilateral_sigma_color_spin.setProperty("class", "img-spin")
        self.bilateral_sigma_color_spin.setRange(10.0, 200.0)
        self.bilateral_sigma_color_spin.setValue(75.0)
        bilateral_layout.addRow("Sigma Color:", self.bilateral_sigma_color_spin)
        
        self.bilateral_sigma_space_spin = QDoubleSpinBox()
        self.bilateral_sigma_space_spin.setProperty("class", "img-spin")
        self.bilateral_sigma_space_spin.setRange(10.0, 200.0)
        self.bilateral_sigma_space_spin.setValue(75.0)
        bilateral_layout.addRow("Sigma Espacio:", self.bilateral_sigma_space_spin)
        
        noise_layout.addWidget(bilateral_frame)
        self.bilateral_frame = bilateral_frame
        
        layout.addWidget(noise_group)
        
        # Edge Enhancement
        edge_group = QGroupBox("Realce de Bordes")
        edge_group.setProperty("class", "img-section")
        edge_layout = QVBoxLayout(edge_group)
        
        # Unsharp mask
        self.unsharp_mask_cb = QCheckBox("Máscara de Desenfoque")
        self.unsharp_mask_cb.setProperty("class", "img-option")
        edge_layout.addWidget(self.unsharp_mask_cb)
        
        unsharp_frame = QFrame()
        unsharp_frame.setEnabled(False)
        unsharp_layout = QFormLayout(unsharp_frame)
        
        self.unsharp_amount_spin = QDoubleSpinBox()
        self.unsharp_amount_spin.setProperty("class", "img-spin")
        self.unsharp_amount_spin.setRange(0.1, 3.0)
        self.unsharp_amount_spin.setSingleStep(0.1)
        self.unsharp_amount_spin.setValue(1.5)
        unsharp_layout.addRow("Cantidad:", self.unsharp_amount_spin)
        
        self.unsharp_radius_spin = QDoubleSpinBox()
        self.unsharp_radius_spin.setProperty("class", "img-spin")
        self.unsharp_radius_spin.setRange(0.5, 5.0)
        self.unsharp_radius_spin.setSingleStep(0.1)
        self.unsharp_radius_spin.setValue(1.0)
        unsharp_layout.addRow("Radio:", self.unsharp_radius_spin)
        
        self.unsharp_threshold_spin = QSpinBox()
        self.unsharp_threshold_spin.setProperty("class", "img-spin")
        self.unsharp_threshold_spin.setRange(0, 255)
        self.unsharp_threshold_spin.setValue(3)
        unsharp_layout.addRow("Umbral:", self.unsharp_threshold_spin)
        
        edge_layout.addWidget(unsharp_frame)
        self.unsharp_frame = unsharp_frame
        
        # Laplacian sharpening
        self.laplacian_cb = QCheckBox("Realce Laplaciano")
        self.laplacian_cb.setProperty("class", "img-option")
        edge_layout.addWidget(self.laplacian_cb)
        
        laplacian_frame = QFrame()
        laplacian_frame.setEnabled(False)
        laplacian_layout = QFormLayout(laplacian_frame)
        
        self.laplacian_strength_spin = QDoubleSpinBox()
        self.laplacian_strength_spin.setProperty("class", "img-spin")
        self.laplacian_strength_spin.setRange(0.1, 2.0)
        self.laplacian_strength_spin.setSingleStep(0.1)
        self.laplacian_strength_spin.setValue(0.5)
        laplacian_layout.addRow("Intensidad:", self.laplacian_strength_spin)
        
        edge_layout.addWidget(laplacian_frame)
        self.laplacian_frame = laplacian_frame
        
        layout.addWidget(edge_group)
        layout.addStretch()
        return widget
        
    def create_morphology_tab(self) -> QWidget:
        """Create morphological operations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Basic Operations
        basic_group = QGroupBox("Operaciones Básicas")
        basic_group.setProperty("class", "img-section")
        basic_layout = QVBoxLayout(basic_group)
        
        # Erosion
        self.erosion_cb = QCheckBox("Erosión")
        self.erosion_cb.setProperty("class", "img-option")
        basic_layout.addWidget(self.erosion_cb)
        
        erosion_frame = QFrame()
        erosion_frame.setEnabled(False)
        erosion_layout = QFormLayout(erosion_frame)
        
        self.erosion_kernel_spin = QSpinBox()
        self.erosion_kernel_spin.setProperty("class", "img-spin")
        self.erosion_kernel_spin.setRange(3, 15)
        self.erosion_kernel_spin.setSingleStep(2)
        self.erosion_kernel_spin.setValue(5)
        erosion_layout.addRow("Tamaño de Kernel:", self.erosion_kernel_spin)
        
        self.erosion_iterations_spin = QSpinBox()
        self.erosion_iterations_spin.setProperty("class", "img-spin")
        self.erosion_iterations_spin.setRange(1, 10)
        self.erosion_iterations_spin.setValue(1)
        erosion_layout.addRow("Iteraciones:", self.erosion_iterations_spin)
        
        basic_layout.addWidget(erosion_frame)
        self.erosion_frame = erosion_frame
        
        # Dilation
        self.dilation_cb = QCheckBox("Dilatación")
        self.dilation_cb.setProperty("class", "img-option")
        basic_layout.addWidget(self.dilation_cb)
        
        dilation_frame = QFrame()
        dilation_frame.setEnabled(False)
        dilation_layout = QFormLayout(dilation_frame)
        
        self.dilation_kernel_spin = QSpinBox()
        self.dilation_kernel_spin.setProperty("class", "img-spin")
        self.dilation_kernel_spin.setRange(3, 15)
        self.dilation_kernel_spin.setSingleStep(2)
        self.dilation_kernel_spin.setValue(5)
        dilation_layout.addRow("Tamaño de Kernel:", self.dilation_kernel_spin)
        
        self.dilation_iterations_spin = QSpinBox()
        self.dilation_iterations_spin.setProperty("class", "img-spin")
        self.dilation_iterations_spin.setRange(1, 10)
        self.dilation_iterations_spin.setValue(1)
        dilation_layout.addRow("Iteraciones:", self.dilation_iterations_spin)
        
        basic_layout.addWidget(dilation_frame)
        self.dilation_frame = dilation_frame
        
        layout.addWidget(basic_group)
        
        # Advanced Operations
        advanced_group = QGroupBox("Operaciones Avanzadas")
        advanced_group.setProperty("class", "img-section")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Opening
        self.opening_cb = QCheckBox("Apertura (Erosión + Dilatación)")
        self.opening_cb.setProperty("class", "img-option")
        advanced_layout.addWidget(self.opening_cb)
        
        # Closing
        self.closing_cb = QCheckBox("Cierre (Dilatación + Erosión)")
        self.closing_cb.setProperty("class", "img-option")
        advanced_layout.addWidget(self.closing_cb)
        
        # Gradient
        self.gradient_cb = QCheckBox("Gradiente Morfológico")
        self.gradient_cb.setProperty("class", "img-option")
        advanced_layout.addWidget(self.gradient_cb)
        
        # Top Hat
        self.tophat_cb = QCheckBox("Top Hat")
        self.tophat_cb.setProperty("class", "img-option")
        advanced_layout.addWidget(self.tophat_cb)
        
        # Black Hat
        self.blackhat_cb = QCheckBox("Black Hat")
        self.blackhat_cb.setProperty("class", "img-option")
        advanced_layout.addWidget(self.blackhat_cb)
        
        layout.addWidget(advanced_group)
        layout.addStretch()
        return widget
        
    def create_advanced_tab(self) -> QWidget:
        """Create advanced processing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Frequency Domain
        frequency_group = QGroupBox("Dominio de Frecuencia")
        frequency_group.setProperty("class", "img-section")
        frequency_layout = QVBoxLayout(frequency_group)
        
        self.fft_filter_cb = QCheckBox("Filtrado FFT")
        self.fft_filter_cb.setProperty("class", "img-option")
        frequency_layout.addWidget(self.fft_filter_cb)
        
        fft_frame = QFrame()
        fft_frame.setEnabled(False)
        fft_layout = QFormLayout(fft_frame)
        
        self.fft_type_combo = QComboBox()
        self.fft_type_combo.setProperty("class", "img-combo")
        self.fft_type_combo.addItems([
            "Pasa Bajas",
            "Pasa Altas", 
            "Pasa Banda",
            "Rechaza Banda"
        ])
        fft_layout.addRow("Tipo de Filtro:", self.fft_type_combo)
        
        self.fft_cutoff_spin = QDoubleSpinBox()
        self.fft_cutoff_spin.setProperty("class", "img-spin")
        self.fft_cutoff_spin.setRange(0.01, 0.5)
        self.fft_cutoff_spin.setSingleStep(0.01)
        self.fft_cutoff_spin.setValue(0.1)
        fft_layout.addRow("Frecuencia de Corte:", self.fft_cutoff_spin)
        
        frequency_layout.addWidget(fft_frame)
        self.fft_frame = fft_frame
        
        layout.addWidget(frequency_group)
        
        # Color Space
        color_group = QGroupBox("Espacio de Color")
        color_group.setProperty("class", "img-section")
        color_layout = QVBoxLayout(color_group)
        
        self.color_conversion_cb = QCheckBox("Conversión de espacio de color")
        self.color_conversion_cb.setProperty("class", "img-option")
        color_layout.addWidget(self.color_conversion_cb)
        
        color_frame = QFrame()
        color_frame.setEnabled(False)
        color_frame_layout = QFormLayout(color_frame)
        
        self.color_space_combo = QComboBox()
        self.color_space_combo.setProperty("class", "img-combo")
        self.color_space_combo.addItems([
            "RGB a HSV",
            "RGB a LAB",
            "RGB a YUV",
            "RGB a Escala de Grises"
        ])
        color_frame_layout.addRow("Conversión:", self.color_space_combo)
        
        color_layout.addWidget(color_frame)
        self.color_frame = color_frame
        
        layout.addWidget(color_group)
        
        # Custom Processing
        custom_group = QGroupBox("Procesamiento Personalizado")
        custom_group.setProperty("class", "img-section")
        custom_layout = QVBoxLayout(custom_group)
        
        self.custom_processing_cb = QCheckBox("Activar procesamiento personalizado")
        self.custom_processing_cb.setProperty("class", "img-option")
        custom_layout.addWidget(self.custom_processing_cb)
        
        custom_frame = QFrame()
        custom_frame.setEnabled(False)
        custom_frame_layout = QVBoxLayout(custom_frame)
        
        custom_label = QLabel("Código Python personalizado:")
        custom_label.setProperty("class", "img-label")
        custom_frame_layout.addWidget(custom_label)
        
        self.custom_code_edit = QTextEdit()
        self.custom_code_edit.setProperty("class", "img-code-edit")
        self.custom_code_edit.setMaximumHeight(120)
        self.custom_code_edit.setPlaceholderText(
            "# Ejemplo:\n"
            "# import cv2\n"
            "# import numpy as np\n"
            "# \n"
            "# def process_image(image):\n"
            "#     # Su código aquí\n"
            "#     return processed_image"
        )
        custom_frame_layout.addWidget(self.custom_code_edit)
        
        custom_layout.addWidget(custom_frame)
        self.custom_frame = custom_frame
        
        layout.addWidget(custom_group)
        layout.addStretch()
        return widget
        
    def connect_signals(self):
        """Connect widget signals"""
        self.enable_processing_cb.toggled.connect(self.main_panel.setEnabled)
        self.enable_processing_cb.toggled.connect(self.emit_configuration_changed)
        
        # Enhancement tab signals
        self.brightness_slider.valueChanged.connect(self.update_brightness_label)
        self.contrast_slider.valueChanged.connect(self.update_contrast_label)
        self.gamma_slider.valueChanged.connect(self.update_gamma_label)
        
        self.gamma_enable_cb.toggled.connect(self.gamma_frame.setEnabled)
        self.clahe_cb.toggled.connect(self.clahe_frame.setEnabled)
        
        # Filtering tab signals
        self.gaussian_blur_cb.toggled.connect(self.gaussian_frame.setEnabled)
        self.median_filter_cb.toggled.connect(self.median_frame.setEnabled)
        self.bilateral_cb.toggled.connect(self.bilateral_frame.setEnabled)
        self.unsharp_mask_cb.toggled.connect(self.unsharp_frame.setEnabled)
        self.laplacian_cb.toggled.connect(self.laplacian_frame.setEnabled)
        
        # Morphology tab signals
        self.erosion_cb.toggled.connect(self.erosion_frame.setEnabled)
        self.dilation_cb.toggled.connect(self.dilation_frame.setEnabled)
        
        # Advanced tab signals
        self.fft_filter_cb.toggled.connect(self.fft_frame.setEnabled)
        self.color_conversion_cb.toggled.connect(self.color_frame.setEnabled)
        self.custom_processing_cb.toggled.connect(self.custom_frame.setEnabled)
        
        # Button signals
        self.preview_btn.clicked.connect(self.request_preview)
        self.reset_btn.clicked.connect(self.reset_configuration)
        self.apply_btn.clicked.connect(self.emit_configuration_changed)
        
        # Connect all widgets to configuration changed
        all_widgets = [
            # Enhancement
            self.brightness_slider, self.contrast_slider, self.gamma_enable_cb,
            self.gamma_slider, self.histogram_eq_cb, self.clahe_cb,
            self.clahe_clip_spin, self.clahe_grid_spin,
            # Filtering
            self.gaussian_blur_cb, self.gaussian_kernel_spin, self.gaussian_sigma_spin,
            self.median_filter_cb, self.median_kernel_spin, self.bilateral_cb,
            self.bilateral_d_spin, self.bilateral_sigma_color_spin, self.bilateral_sigma_space_spin,
            self.unsharp_mask_cb, self.unsharp_amount_spin, self.unsharp_radius_spin,
            self.unsharp_threshold_spin, self.laplacian_cb, self.laplacian_strength_spin,
            # Morphology
            self.erosion_cb, self.erosion_kernel_spin, self.erosion_iterations_spin,
            self.dilation_cb, self.dilation_kernel_spin, self.dilation_iterations_spin,
            self.opening_cb, self.closing_cb, self.gradient_cb, self.tophat_cb, self.blackhat_cb,
            # Advanced
            self.fft_filter_cb, self.fft_type_combo, self.fft_cutoff_spin,
            self.color_conversion_cb, self.color_space_combo,
            self.custom_processing_cb, self.custom_code_edit
        ]
        
        for widget in all_widgets:
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'textChanged'):
                widget.textChanged.connect(self.emit_configuration_changed)
                
    def update_brightness_label(self, value):
        """Update brightness label"""
        self.brightness_value_label.setText(str(value))
        
    def update_contrast_label(self, value):
        """Update contrast label"""
        self.contrast_value_label.setText(str(value))
        
    def update_gamma_label(self, value):
        """Update gamma label"""
        gamma = value / 100.0
        self.gamma_value_label.setText(f"{gamma:.1f}")
        
    def request_preview(self):
        """Request preview with current configuration"""
        config = self.get_configuration()
        self.previewRequested.emit(config)
        
    def reset_configuration(self):
        """Reset all configuration to defaults"""
        # Enhancement
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.gamma_enable_cb.setChecked(False)
        self.gamma_slider.setValue(100)
        self.histogram_eq_cb.setChecked(False)
        self.clahe_cb.setChecked(False)
        
        # Filtering
        self.gaussian_blur_cb.setChecked(False)
        self.median_filter_cb.setChecked(False)
        self.bilateral_cb.setChecked(False)
        self.unsharp_mask_cb.setChecked(False)
        self.laplacian_cb.setChecked(False)
        
        # Morphology
        self.erosion_cb.setChecked(False)
        self.dilation_cb.setChecked(False)
        self.opening_cb.setChecked(False)
        self.closing_cb.setChecked(False)
        self.gradient_cb.setChecked(False)
        self.tophat_cb.setChecked(False)
        self.blackhat_cb.setChecked(False)
        
        # Advanced
        self.fft_filter_cb.setChecked(False)
        self.color_conversion_cb.setChecked(False)
        self.custom_processing_cb.setChecked(False)
        self.custom_code_edit.clear()
        
        self.emit_configuration_changed()
        
    def emit_configuration_changed(self):
        """Emit configuration changed signal"""
        self.config = self.get_configuration()
        self.configurationChanged.emit(self.config)
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get current image processing configuration"""
        config = {
            'enabled': self.enable_processing_cb.isChecked(),
            'enhancement': {
                'brightness': self.brightness_slider.value(),
                'contrast': self.contrast_slider.value(),
                'gamma': {
                    'enabled': self.gamma_enable_cb.isChecked(),
                    'value': self.gamma_slider.value() / 100.0
                },
                'histogram_equalization': self.histogram_eq_cb.isChecked(),
                'clahe': {
                    'enabled': self.clahe_cb.isChecked(),
                    'clip_limit': self.clahe_clip_spin.value(),
                    'grid_size': self.clahe_grid_spin.value()
                }
            },
            'filtering': {
                'gaussian_blur': {
                    'enabled': self.gaussian_blur_cb.isChecked(),
                    'kernel_size': self.gaussian_kernel_spin.value(),
                    'sigma': self.gaussian_sigma_spin.value()
                },
                'median_filter': {
                    'enabled': self.median_filter_cb.isChecked(),
                    'kernel_size': self.median_kernel_spin.value()
                },
                'bilateral': {
                    'enabled': self.bilateral_cb.isChecked(),
                    'd': self.bilateral_d_spin.value(),
                    'sigma_color': self.bilateral_sigma_color_spin.value(),
                    'sigma_space': self.bilateral_sigma_space_spin.value()
                },
                'unsharp_mask': {
                    'enabled': self.unsharp_mask_cb.isChecked(),
                    'amount': self.unsharp_amount_spin.value(),
                    'radius': self.unsharp_radius_spin.value(),
                    'threshold': self.unsharp_threshold_spin.value()
                },
                'laplacian': {
                    'enabled': self.laplacian_cb.isChecked(),
                    'strength': self.laplacian_strength_spin.value()
                }
            },
            'morphology': {
                'erosion': {
                    'enabled': self.erosion_cb.isChecked(),
                    'kernel_size': self.erosion_kernel_spin.value(),
                    'iterations': self.erosion_iterations_spin.value()
                },
                'dilation': {
                    'enabled': self.dilation_cb.isChecked(),
                    'kernel_size': self.dilation_kernel_spin.value(),
                    'iterations': self.dilation_iterations_spin.value()
                },
                'opening': self.opening_cb.isChecked(),
                'closing': self.closing_cb.isChecked(),
                'gradient': self.gradient_cb.isChecked(),
                'tophat': self.tophat_cb.isChecked(),
                'blackhat': self.blackhat_cb.isChecked()
            },
            'advanced': {
                'fft_filter': {
                    'enabled': self.fft_filter_cb.isChecked(),
                    'type': self.fft_type_combo.currentText(),
                    'cutoff': self.fft_cutoff_spin.value()
                },
                'color_conversion': {
                    'enabled': self.color_conversion_cb.isChecked(),
                    'target_space': self.color_space_combo.currentText()
                },
                'custom_processing': {
                    'enabled': self.custom_processing_cb.isChecked(),
                    'code': self.custom_code_edit.toPlainText()
                }
            }
        }
        
        return config
        
    def set_configuration(self, config: Dict[str, Any]):
        """Set image processing configuration"""
        self.enable_processing_cb.setChecked(config.get('enabled', False))
        
        # Enhancement
        enhancement = config.get('enhancement', {})
        self.brightness_slider.setValue(enhancement.get('brightness', 0))
        self.contrast_slider.setValue(enhancement.get('contrast', 0))
        
        gamma = enhancement.get('gamma', {})
        self.gamma_enable_cb.setChecked(gamma.get('enabled', False))
        self.gamma_slider.setValue(int(gamma.get('value', 1.0) * 100))
        
        self.histogram_eq_cb.setChecked(enhancement.get('histogram_equalization', False))
        
        clahe = enhancement.get('clahe', {})
        self.clahe_cb.setChecked(clahe.get('enabled', False))
        self.clahe_clip_spin.setValue(clahe.get('clip_limit', 2.0))
        self.clahe_grid_spin.setValue(clahe.get('grid_size', 8))
        
        # Filtering
        filtering = config.get('filtering', {})
        
        gaussian = filtering.get('gaussian_blur', {})
        self.gaussian_blur_cb.setChecked(gaussian.get('enabled', False))
        self.gaussian_kernel_spin.setValue(gaussian.get('kernel_size', 5))
        self.gaussian_sigma_spin.setValue(gaussian.get('sigma', 1.0))
        
        median = filtering.get('median_filter', {})
        self.median_filter_cb.setChecked(median.get('enabled', False))
        self.median_kernel_spin.setValue(median.get('kernel_size', 5))
        
        bilateral = filtering.get('bilateral', {})
        self.bilateral_cb.setChecked(bilateral.get('enabled', False))
        self.bilateral_d_spin.setValue(bilateral.get('d', 9))
        self.bilateral_sigma_color_spin.setValue(bilateral.get('sigma_color', 75.0))
        self.bilateral_sigma_space_spin.setValue(bilateral.get('sigma_space', 75.0))
        
        unsharp = filtering.get('unsharp_mask', {})
        self.unsharp_mask_cb.setChecked(unsharp.get('enabled', False))
        self.unsharp_amount_spin.setValue(unsharp.get('amount', 1.5))
        self.unsharp_radius_spin.setValue(unsharp.get('radius', 1.0))
        self.unsharp_threshold_spin.setValue(unsharp.get('threshold', 3))
        
        laplacian = filtering.get('laplacian', {})
        self.laplacian_cb.setChecked(laplacian.get('enabled', False))
        self.laplacian_strength_spin.setValue(laplacian.get('strength', 0.5))
        
        # Morphology
        morphology = config.get('morphology', {})
        
        erosion = morphology.get('erosion', {})
        self.erosion_cb.setChecked(erosion.get('enabled', False))
        self.erosion_kernel_spin.setValue(erosion.get('kernel_size', 5))
        self.erosion_iterations_spin.setValue(erosion.get('iterations', 1))
        
        dilation = morphology.get('dilation', {})
        self.dilation_cb.setChecked(dilation.get('enabled', False))
        self.dilation_kernel_spin.setValue(dilation.get('kernel_size', 5))
        self.dilation_iterations_spin.setValue(dilation.get('iterations', 1))
        
        self.opening_cb.setChecked(morphology.get('opening', False))
        self.closing_cb.setChecked(morphology.get('closing', False))
        self.gradient_cb.setChecked(morphology.get('gradient', False))
        self.tophat_cb.setChecked(morphology.get('tophat', False))
        self.blackhat_cb.setChecked(morphology.get('blackhat', False))
        
        # Advanced
        advanced = config.get('advanced', {})
        
        fft = advanced.get('fft_filter', {})
        self.fft_filter_cb.setChecked(fft.get('enabled', False))
        fft_type = fft.get('type', '')
        if fft_type:
            index = self.fft_type_combo.findText(fft_type)
            if index >= 0:
                self.fft_type_combo.setCurrentIndex(index)
        self.fft_cutoff_spin.setValue(fft.get('cutoff', 0.1))
        
        color = advanced.get('color_conversion', {})
        self.color_conversion_cb.setChecked(color.get('enabled', False))
        color_space = color.get('target_space', '')
        if color_space:
            index = self.color_space_combo.findText(color_space)
            if index >= 0:
                self.color_space_combo.setCurrentIndex(index)
                
        custom = advanced.get('custom_processing', {})
        self.custom_processing_cb.setChecked(custom.get('enabled', False))
        self.custom_code_edit.setPlainText(custom.get('code', ''))
        
    def is_enabled(self) -> bool:
        """Check if image processing is enabled"""
        return self.enable_processing_cb.isChecked()
        
    def validate_configuration(self) -> tuple[bool, list]:
        """Validate current configuration"""
        errors = []
        
        if not self.is_enabled():
            return True, []
            
        # Validate custom code if enabled
        if self.custom_processing_cb.isChecked():
            code = self.custom_code_edit.toPlainText().strip()
            if not code:
                errors.append("Código personalizado está vacío")
            elif 'def process_image' not in code:
                errors.append("Código personalizado debe contener función 'process_image'")
                
        return len(errors) == 0, errors

    def set_configuration_level(self, level: str):
        """Set default options according to the selected configuration level.
        Accepted values (case-insensitive): 'Basic', 'Intermediate', 'Advanced'.
        """
        lvl = (level or '').strip().lower()
        if lvl == 'basic':
            # Enable processing with minimal, safe defaults
            self.enable_processing_cb.setChecked(True)
            # Enhancement: keep simple
            self.gamma_enable_cb.setChecked(False)
            self.histogram_eq_cb.setChecked(False)
            self.clahe_cb.setChecked(False)
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)
            self.gamma_slider.setValue(100)
            # Filtering: minimal blur only
            self.gaussian_blur_cb.setChecked(True)
            self.gaussian_kernel_spin.setValue(max(3, self.gaussian_kernel_spin.value()))
            self.gaussian_sigma_spin.setValue(max(1.0, self.gaussian_sigma_spin.value()))
            self.median_filter_cb.setChecked(False)
            self.bilateral_cb.setChecked(False)
            self.unsharp_mask_cb.setChecked(False)
            self.laplacian_cb.setChecked(False)
            # Morphology: off by default
            self.erosion_cb.setChecked(False)
            self.dilation_cb.setChecked(False)
            self.opening_cb.setChecked(False)
            self.closing_cb.setChecked(False)
            self.gradient_cb.setChecked(False)
            self.tophat_cb.setChecked(False)
            self.blackhat_cb.setChecked(False)
            # Advanced: disabled
            self.fft_filter_cb.setChecked(False)
            self.color_conversion_cb.setChecked(False)
            self.custom_processing_cb.setChecked(False)
        elif lvl == 'intermediate':
            # Balanced defaults for typical comparison workflows
            self.enable_processing_cb.setChecked(True)
            # Enhancement
            self.gamma_enable_cb.setChecked(True)
            self.histogram_eq_cb.setChecked(True)
            self.clahe_cb.setChecked(True)
            # Filtering
            self.gaussian_blur_cb.setChecked(True)
            self.median_filter_cb.setChecked(True)
            self.bilateral_cb.setChecked(False)
            self.unsharp_mask_cb.setChecked(True)
            self.laplacian_cb.setChecked(False)
            # Morphology: enable basic opening/closing pipeline
            self.erosion_cb.setChecked(True)
            self.dilation_cb.setChecked(True)
            self.opening_cb.setChecked(True)
            self.closing_cb.setChecked(False)
            self.gradient_cb.setChecked(False)
            self.tophat_cb.setChecked(False)
            self.blackhat_cb.setChecked(False)
            # Advanced
            self.fft_filter_cb.setChecked(False)
            self.color_conversion_cb.setChecked(True)
            self.custom_processing_cb.setChecked(False)
        else:
            # Advanced: enable comprehensive options
            self.enable_processing_cb.setChecked(True)
            # Enhancement
            self.gamma_enable_cb.setChecked(True)
            self.histogram_eq_cb.setChecked(True)
            self.clahe_cb.setChecked(True)
            # Filtering
            self.gaussian_blur_cb.setChecked(True)
            self.median_filter_cb.setChecked(True)
            self.bilateral_cb.setChecked(True)
            self.unsharp_mask_cb.setChecked(True)
            self.laplacian_cb.setChecked(True)
            # Morphology: full set
            self.erosion_cb.setChecked(True)
            self.dilation_cb.setChecked(True)
            self.opening_cb.setChecked(True)
            self.closing_cb.setChecked(True)
            self.gradient_cb.setChecked(True)
            self.tophat_cb.setChecked(True)
            self.blackhat_cb.setChecked(True)
            # Advanced tools
            self.fft_filter_cb.setChecked(True)
            self.color_conversion_cb.setChecked(True)
            # Keep custom processing off by default to avoid code execution surprises
            self.custom_processing_cb.setChecked(False)
        # Emit updated configuration
        self.emit_configuration_changed()