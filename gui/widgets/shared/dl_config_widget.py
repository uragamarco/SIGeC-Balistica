"""
DeepLearningConfigWidget - Shared Deep Learning configuration component
Eliminates code duplication between analysis and comparison tabs
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QCheckBox, QRadioButton, QSlider,
                             QLabel, QFrame, QButtonGroup, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTextEdit, QPushButton,
                             QProgressBar, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon
from typing import Dict, Any, List


class DeepLearningConfigWidget(QWidget):
    """Shared Deep Learning configuration widget for ballistic analysis"""
    
    configurationChanged = pyqtSignal(dict)  # Emitted when configuration changes
    modelLoadRequested = pyqtSignal(str)     # Emitted when model load is requested
    
    def __init__(self, mode: str = "analysis", parent=None):
        """
        Initialize Deep Learning configuration widget
        
        Args:
            mode: "analysis" or "comparison" to customize labels and options
        """
        super().__init__(parent)
        self.mode = mode
        self.config = {}
        self.available_models = []
        
        self.setup_ui()
        self.connect_signals()
        self.load_available_models()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Main enable checkbox
        self.enable_dl_cb = QCheckBox(
            "Activar análisis con Deep Learning" + 
            (" para comparación" if self.mode == "comparison" else " para análisis")
        )
        self.enable_dl_cb.setProperty("class", "dl-enable-checkbox")
        layout.addWidget(self.enable_dl_cb)
        
        # Main configuration panel
        self.main_panel = QFrame()
        self.main_panel.setProperty("class", "dl-config-panel")
        self.main_panel.setEnabled(False)
        
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)
        
        # Model Selection Section
        model_section = self.create_model_section()
        main_layout.addWidget(model_section)
        
        # Analysis Configuration Section
        analysis_section = self.create_analysis_section()
        main_layout.addWidget(analysis_section)
        
        # Performance Section
        performance_section = self.create_performance_section()
        main_layout.addWidget(performance_section)
        
        # Advanced Options Section
        advanced_section = self.create_advanced_section()
        main_layout.addWidget(advanced_section)
        
        layout.addWidget(self.main_panel)
        
    def create_model_section(self) -> QWidget:
        """Create model selection section"""
        group = QGroupBox("Selección de Modelo Deep Learning")
        group.setProperty("class", "dl-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Model selection
        model_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.setProperty("class", "dl-combo")
        model_layout.addRow("Modelo Principal:", self.model_combo)
        
        # Model info display
        self.model_info_label = QLabel("Seleccione un modelo para ver información")
        self.model_info_label.setProperty("class", "dl-model-info")
        self.model_info_label.setWordWrap(True)
        model_layout.addRow("Información:", self.model_info_label)
        
        layout.addLayout(model_layout)
        
        # Model management buttons
        button_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("Cargar Modelo")
        self.load_model_btn.setProperty("class", "dl-button primary")
        self.load_model_btn.setEnabled(False)
        
        self.refresh_models_btn = QPushButton("Actualizar Lista")
        self.refresh_models_btn.setProperty("class", "dl-button secondary")
        
        self.model_config_btn = QPushButton("Configuración Avanzada")
        self.model_config_btn.setProperty("class", "dl-button secondary")
        
        button_layout.addWidget(self.load_model_btn)
        button_layout.addWidget(self.refresh_models_btn)
        button_layout.addWidget(self.model_config_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Model loading progress
        self.model_progress = QProgressBar()
        self.model_progress.setProperty("class", "dl-progress")
        self.model_progress.setVisible(False)
        layout.addWidget(self.model_progress)
        
        return group
        
    def create_analysis_section(self) -> QWidget:
        """Create analysis configuration section"""
        group = QGroupBox("Configuración de Análisis")
        group.setProperty("class", "dl-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Confidence threshold
        confidence_frame = QFrame()
        confidence_layout = QVBoxLayout(confidence_frame)
        
        confidence_label = QLabel("Umbral de Confianza")
        confidence_label.setProperty("class", "dl-label")
        confidence_layout.addWidget(confidence_label)
        
        # Slider with value display
        slider_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setProperty("class", "dl-slider")
        self.confidence_slider.setRange(50, 99)
        self.confidence_slider.setValue(85)
        
        self.confidence_value_label = QLabel("0.85")
        self.confidence_value_label.setProperty("class", "dl-value-label")
        self.confidence_value_label.setMinimumWidth(40)
        
        slider_layout.addWidget(self.confidence_slider)
        slider_layout.addWidget(self.confidence_value_label)
        
        confidence_layout.addLayout(slider_layout)
        
        # Confidence description
        conf_desc = QLabel("Valor recomendado: 0.80-0.90 para análisis balístico")
        conf_desc.setProperty("class", "dl-description")
        confidence_layout.addWidget(conf_desc)
        
        layout.addWidget(confidence_frame)
        
        # Analysis features
        features_frame = QFrame()
        features_layout = QVBoxLayout(features_frame)
        
        features_label = QLabel("Características a Analizar:")
        features_label.setProperty("class", "dl-label")
        features_layout.addWidget(features_label)
        
        # Feature checkboxes
        self.firing_pin_cb = QCheckBox("Impresión de percutor")
        self.firing_pin_cb.setProperty("class", "dl-feature")
        self.firing_pin_cb.setChecked(True)
        features_layout.addWidget(self.firing_pin_cb)
        
        self.breech_face_cb = QCheckBox("Cara de recámara")
        self.breech_face_cb.setProperty("class", "dl-feature")
        self.breech_face_cb.setChecked(True)
        features_layout.addWidget(self.breech_face_cb)
        
        self.extractor_cb = QCheckBox("Marcas de extractor")
        self.extractor_cb.setProperty("class", "dl-feature")
        self.extractor_cb.setChecked(True)
        features_layout.addWidget(self.extractor_cb)
        
        self.ejector_cb = QCheckBox("Marcas de eyector")
        self.ejector_cb.setProperty("class", "dl-feature")
        self.ejector_cb.setChecked(True)
        features_layout.addWidget(self.ejector_cb)
        
        self.striations_cb = QCheckBox("Estrías y campos")
        self.striations_cb.setProperty("class", "dl-feature")
        self.striations_cb.setChecked(True)
        features_layout.addWidget(self.striations_cb)
        
        layout.addWidget(features_frame)
        
        return group
        
    def create_performance_section(self) -> QWidget:
        """Create performance configuration section"""
        group = QGroupBox("Configuración de Rendimiento")
        group.setProperty("class", "dl-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Processing mode
        mode_layout = QFormLayout()
        
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.setProperty("class", "dl-combo")
        self.processing_mode_combo.addItems([
            "CPU - Procesamiento en CPU (Lento, Compatible)",
            "GPU - Procesamiento en GPU (Rápido, Requiere CUDA)",
            "Automático - Selección automática según hardware",
            "Híbrido - Combinación CPU/GPU optimizada"
        ])
        self.processing_mode_combo.setCurrentIndex(2)  # Automatic by default
        mode_layout.addRow("Modo de Procesamiento:", self.processing_mode_combo)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setProperty("class", "dl-spin")
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        mode_layout.addRow("Tamaño de Lote:", self.batch_size_spin)
        
        # Number of workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setProperty("class", "dl-spin")
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        mode_layout.addRow("Hilos de Procesamiento:", self.workers_spin)
        
        layout.addLayout(mode_layout)
        
        # Memory optimization
        memory_frame = QFrame()
        memory_layout = QVBoxLayout(memory_frame)
        
        self.memory_optimization_cb = QCheckBox("Optimización de memoria")
        self.memory_optimization_cb.setProperty("class", "dl-option")
        self.memory_optimization_cb.setChecked(True)
        memory_layout.addWidget(self.memory_optimization_cb)
        
        self.gradient_checkpointing_cb = QCheckBox("Gradient checkpointing (reduce memoria)")
        self.gradient_checkpointing_cb.setProperty("class", "dl-option")
        memory_layout.addWidget(self.gradient_checkpointing_cb)
        
        layout.addWidget(memory_frame)
        
        return group
        
    def create_advanced_section(self) -> QWidget:
        """Create advanced options section"""
        group = QGroupBox("Opciones Avanzadas")
        group.setProperty("class", "dl-section")
        
        layout = QVBoxLayout(group)
        layout.setSpacing(15)
        
        # Ensemble methods
        ensemble_frame = QFrame()
        ensemble_layout = QVBoxLayout(ensemble_frame)
        
        self.ensemble_cb = QCheckBox("Usar ensemble de modelos")
        self.ensemble_cb.setProperty("class", "dl-option")
        ensemble_layout.addWidget(self.ensemble_cb)
        
        # Ensemble models list
        self.ensemble_list = QListWidget()
        self.ensemble_list.setProperty("class", "dl-ensemble-list")
        self.ensemble_list.setMaximumHeight(100)
        self.ensemble_list.setEnabled(False)
        ensemble_layout.addWidget(self.ensemble_list)
        
        layout.addWidget(ensemble_frame)
        
        # Data augmentation
        augmentation_frame = QFrame()
        augmentation_layout = QVBoxLayout(augmentation_frame)
        
        augmentation_label = QLabel("Aumento de Datos:")
        augmentation_label.setProperty("class", "dl-label")
        augmentation_layout.addWidget(augmentation_label)
        
        self.rotation_cb = QCheckBox("Rotación")
        self.rotation_cb.setProperty("class", "dl-augmentation")
        augmentation_layout.addWidget(self.rotation_cb)
        
        self.scaling_cb = QCheckBox("Escalado")
        self.scaling_cb.setProperty("class", "dl-augmentation")
        augmentation_layout.addWidget(self.scaling_cb)
        
        self.noise_cb = QCheckBox("Ruido gaussiano")
        self.noise_cb.setProperty("class", "dl-augmentation")
        augmentation_layout.addWidget(self.noise_cb)
        
        self.brightness_cb = QCheckBox("Ajuste de brillo")
        self.brightness_cb.setProperty("class", "dl-augmentation")
        augmentation_layout.addWidget(self.brightness_cb)
        
        layout.addWidget(augmentation_frame)
        
        # Output options
        output_frame = QFrame()
        output_layout = QVBoxLayout(output_frame)
        
        self.save_predictions_cb = QCheckBox("Guardar predicciones detalladas")
        self.save_predictions_cb.setProperty("class", "dl-option")
        self.save_predictions_cb.setChecked(True)
        output_layout.addWidget(self.save_predictions_cb)
        
        self.generate_heatmaps_cb = QCheckBox("Generar mapas de calor de atención")
        self.generate_heatmaps_cb.setProperty("class", "dl-option")
        output_layout.addWidget(self.generate_heatmaps_cb)
        
        self.export_features_cb = QCheckBox("Exportar características extraídas")
        self.export_features_cb.setProperty("class", "dl-option")
        output_layout.addWidget(self.export_features_cb)
        
        layout.addWidget(output_frame)
        
        return group
        
    def connect_signals(self):
        """Connect widget signals"""
        self.enable_dl_cb.toggled.connect(self.main_panel.setEnabled)
        self.enable_dl_cb.toggled.connect(self.emit_configuration_changed)
        
        # Model selection
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.model_combo.currentTextChanged.connect(self.emit_configuration_changed)
        
        # Buttons
        self.load_model_btn.clicked.connect(self.load_selected_model)
        self.refresh_models_btn.clicked.connect(self.load_available_models)
        self.model_config_btn.clicked.connect(self.open_model_config)
        
        # Confidence slider
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        self.confidence_slider.valueChanged.connect(self.emit_configuration_changed)
        
        # Ensemble checkbox
        self.ensemble_cb.toggled.connect(self.ensemble_list.setEnabled)
        self.ensemble_cb.toggled.connect(self.emit_configuration_changed)
        
        # All other widgets
        widgets = [
            self.processing_mode_combo, self.batch_size_spin, self.workers_spin,
            self.memory_optimization_cb, self.gradient_checkpointing_cb,
            self.firing_pin_cb, self.breech_face_cb, self.extractor_cb,
            self.ejector_cb, self.striations_cb, self.rotation_cb,
            self.scaling_cb, self.noise_cb, self.brightness_cb,
            self.save_predictions_cb, self.generate_heatmaps_cb, self.export_features_cb
        ]
        
        for widget in widgets:
            if hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self.emit_configuration_changed)
            elif hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self.emit_configuration_changed)
                
    def load_available_models(self):
        """Load available Deep Learning models"""
        # Simulate loading models (in real implementation, this would scan model directory)
        self.available_models = [
            {
                'name': 'BallisticNet-v2.1',
                'description': 'Modelo principal para análisis balístico completo',
                'accuracy': 0.94,
                'size': '245 MB',
                'features': ['firing_pin', 'breech_face', 'extractor', 'ejector']
            },
            {
                'name': 'FirearmID-CNN',
                'description': 'Red convolucional especializada en identificación de armas',
                'accuracy': 0.91,
                'size': '180 MB',
                'features': ['firing_pin', 'breech_face']
            },
            {
                'name': 'StriationNet',
                'description': 'Modelo especializado en análisis de estrías',
                'accuracy': 0.89,
                'size': '120 MB',
                'features': ['striations']
            },
            {
                'name': 'MultiScale-Ballistic',
                'description': 'Análisis multi-escala para características complejas',
                'accuracy': 0.92,
                'size': '320 MB',
                'features': ['firing_pin', 'breech_face', 'extractor', 'ejector', 'striations']
            }
        ]
        
        # Update combo box
        self.model_combo.clear()
        for model in self.available_models:
            self.model_combo.addItem(model['name'])
            
        # Update ensemble list
        self.ensemble_list.clear()
        for model in self.available_models:
            item = QListWidgetItem(model['name'])
            item.setCheckState(Qt.Unchecked)
            self.ensemble_list.addItem(item)
            
    def on_model_changed(self, model_name: str):
        """Handle model selection change"""
        model_info = next((m for m in self.available_models if m['name'] == model_name), None)
        
        if model_info:
            info_text = (f"Precisión: {model_info['accuracy']:.1%} | "
                        f"Tamaño: {model_info['size']} | "
                        f"Características: {', '.join(model_info['features'])}\n"
                        f"{model_info['description']}")
            self.model_info_label.setText(info_text)
            self.load_model_btn.setEnabled(True)
        else:
            self.model_info_label.setText("Seleccione un modelo para ver información")
            self.load_model_btn.setEnabled(False)
            
    def load_selected_model(self):
        """Load the selected model"""
        model_name = self.model_combo.currentText()
        if model_name:
            self.model_progress.setVisible(True)
            self.model_progress.setValue(0)
            self.load_model_btn.setEnabled(False)
            
            # Simulate model loading with progress
            self.load_timer = QTimer()
            self.load_progress = 0
            self.load_timer.timeout.connect(self.update_load_progress)
            self.load_timer.start(100)
            
            self.modelLoadRequested.emit(model_name)
            
    def update_load_progress(self):
        """Update model loading progress"""
        self.load_progress += 5
        self.model_progress.setValue(self.load_progress)
        
        if self.load_progress >= 100:
            self.load_timer.stop()
            self.model_progress.setVisible(False)
            self.load_model_btn.setEnabled(True)
            self.load_model_btn.setText("Modelo Cargado ✓")
            self.load_model_btn.setProperty("class", "dl-button success")
            
    def open_model_config(self):
        """Open advanced model configuration dialog"""
        # This would open a detailed configuration dialog
        pass
        
    def update_confidence_label(self, value):
        """Update confidence label when slider changes"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        
    def emit_configuration_changed(self):
        """Emit configuration changed signal"""
        self.config = self.get_configuration()
        self.configurationChanged.emit(self.config)
        
    def get_configuration(self) -> Dict[str, Any]:
        """Get current Deep Learning configuration"""
        # Get selected ensemble models
        ensemble_models = []
        for i in range(self.ensemble_list.count()):
            item = self.ensemble_list.item(i)
            if item.checkState() == Qt.Checked:
                ensemble_models.append(item.text())
                
        config = {
            'enabled': self.enable_dl_cb.isChecked(),
            'model': self.model_combo.currentText(),
            'confidence_threshold': self.confidence_slider.value() / 100.0,
            'processing_mode': self.processing_mode_combo.currentText(),
            'batch_size': self.batch_size_spin.value(),
            'num_workers': self.workers_spin.value(),
            'memory_optimization': self.memory_optimization_cb.isChecked(),
            'gradient_checkpointing': self.gradient_checkpointing_cb.isChecked(),
            'features': {
                'firing_pin': self.firing_pin_cb.isChecked(),
                'breech_face': self.breech_face_cb.isChecked(),
                'extractor': self.extractor_cb.isChecked(),
                'ejector': self.ejector_cb.isChecked(),
                'striations': self.striations_cb.isChecked()
            },
            'ensemble': {
                'enabled': self.ensemble_cb.isChecked(),
                'models': ensemble_models
            },
            'augmentation': {
                'rotation': self.rotation_cb.isChecked(),
                'scaling': self.scaling_cb.isChecked(),
                'noise': self.noise_cb.isChecked(),
                'brightness': self.brightness_cb.isChecked()
            },
            'output': {
                'save_predictions': self.save_predictions_cb.isChecked(),
                'generate_heatmaps': self.generate_heatmaps_cb.isChecked(),
                'export_features': self.export_features_cb.isChecked()
            }
        }
        
        return config
        
    def set_configuration(self, config: Dict[str, Any]):
        """Set Deep Learning configuration"""
        self.enable_dl_cb.setChecked(config.get('enabled', False))
        
        # Model selection
        model = config.get('model', '')
        if model:
            index = self.model_combo.findText(model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                
        # Confidence threshold
        confidence = config.get('confidence_threshold', 0.85)
        self.confidence_slider.setValue(int(confidence * 100))
        
        # Performance settings
        processing_mode = config.get('processing_mode', '')
        if processing_mode:
            index = self.processing_mode_combo.findText(processing_mode)
            if index >= 0:
                self.processing_mode_combo.setCurrentIndex(index)
                
        self.batch_size_spin.setValue(config.get('batch_size', 8))
        self.workers_spin.setValue(config.get('num_workers', 4))
        self.memory_optimization_cb.setChecked(config.get('memory_optimization', True))
        self.gradient_checkpointing_cb.setChecked(config.get('gradient_checkpointing', False))
        
        # Features
        features = config.get('features', {})
        self.firing_pin_cb.setChecked(features.get('firing_pin', True))
        self.breech_face_cb.setChecked(features.get('breech_face', True))
        self.extractor_cb.setChecked(features.get('extractor', True))
        self.ejector_cb.setChecked(features.get('ejector', True))
        self.striations_cb.setChecked(features.get('striations', True))
        
        # Ensemble
        ensemble = config.get('ensemble', {})
        self.ensemble_cb.setChecked(ensemble.get('enabled', False))
        ensemble_models = ensemble.get('models', [])
        for i in range(self.ensemble_list.count()):
            item = self.ensemble_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in ensemble_models else Qt.Unchecked)
            
        # Augmentation
        augmentation = config.get('augmentation', {})
        self.rotation_cb.setChecked(augmentation.get('rotation', False))
        self.scaling_cb.setChecked(augmentation.get('scaling', False))
        self.noise_cb.setChecked(augmentation.get('noise', False))
        self.brightness_cb.setChecked(augmentation.get('brightness', False))
        
        # Output
        output = config.get('output', {})
        self.save_predictions_cb.setChecked(output.get('save_predictions', True))
        self.generate_heatmaps_cb.setChecked(output.get('generate_heatmaps', False))
        self.export_features_cb.setChecked(output.get('export_features', False))
        
    def is_enabled(self) -> bool:
        """Check if Deep Learning configuration is enabled"""
        return self.enable_dl_cb.isChecked()
        
    def get_selected_model(self) -> str:
        """Get currently selected model"""
        return self.model_combo.currentText()
        
    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold"""
        return self.confidence_slider.value() / 100.0
        
    def validate_configuration(self) -> tuple[bool, list]:
        """Validate current configuration"""
        errors = []
        
        if not self.is_enabled():
            return True, []
            
        # Validate model selection
        if not self.get_selected_model():
            errors.append("Debe seleccionar un modelo de Deep Learning")
            
        # Validate confidence threshold
        confidence = self.get_confidence_threshold()
        if confidence < 0.5 or confidence > 0.99:
            errors.append("Umbral de confianza debe estar entre 0.50 y 0.99")
            
        # Validate batch size
        if self.batch_size_spin.value() < 1:
            errors.append("Tamaño de lote debe ser al menos 1")
            
        # Validate features selection
        features = [
            self.firing_pin_cb.isChecked(),
            self.breech_face_cb.isChecked(),
            self.extractor_cb.isChecked(),
            self.ejector_cb.isChecked(),
            self.striations_cb.isChecked()
        ]
        
        if not any(features):
            errors.append("Debe seleccionar al menos una característica para analizar")
            
        return len(errors) == 0, errors
    
    def set_mode(self, mode: str):
        """Set the widget mode (for compatibility)"""
        self.mode = mode
        # Update UI elements based on mode if needed
        if hasattr(self, 'enable_dl_cb'):
            self.enable_dl_cb.setText(
                "Habilitar análisis con Deep Learning" + 
                (" para comparación" if mode == "comparison" else " para análisis")
            )