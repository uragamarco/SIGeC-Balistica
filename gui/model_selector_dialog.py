#!/usr/bin/env python3
"""
Diálogo de Selección de Modelos de Deep Learning
Sistema SIGeC-Balistica- Análisis de Cartuchos y Balas Automático

Permite seleccionar y configurar modelos de deep learning para análisis balístico.
"""

import os
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QTabWidget, QWidget, QTextEdit,
    QSlider, QFrame, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

# Importaciones condicionales para Deep Learning
try:
    from deep_learning.config.experiment_config import ModelConfig, DataConfig, TrainingConfig
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

class ModelSelectorDialog(QDialog):
    """Diálogo para seleccionar y configurar modelos de deep learning"""
    
    modelConfigured = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_config: Optional[Dict] = None):
        super().__init__(parent)
        self.current_config = current_config or {}
        self.setup_ui()
        self.load_current_config()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        self.setWindowTitle("Configuración de Modelos de Deep Learning")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Configuración de Modelos de Deep Learning")
        header_label.setProperty("class", "title")
        layout.addWidget(header_label)
        
        # Tabs para diferentes aspectos de configuración
        self.tabs = QTabWidget()
        
        # Tab 1: Selección de Modelo
        model_tab = self.create_model_selection_tab()
        self.tabs.addTab(model_tab, "🧠 Modelo")
        
        # Tab 2: Configuración de Datos
        data_tab = self.create_data_config_tab()
        self.tabs.addTab(data_tab, "📊 Datos")
        
        # Tab 3: Parámetros de Entrenamiento
        training_tab = self.create_training_config_tab()
        self.tabs.addTab(training_tab, "⚙️ Entrenamiento")
        
        # Tab 4: Vista Previa de Configuración
        preview_tab = self.create_preview_tab()
        self.tabs.addTab(preview_tab, "👁️ Vista Previa")
        
        layout.addWidget(self.tabs)
        
        # Botones
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("🔄 Restablecer")
        self.reset_button.clicked.connect(self.reset_config)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("❌ Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.apply_button = QPushButton("✅ Aplicar Configuración")
        self.apply_button.setProperty("class", "primary-button")
        self.apply_button.clicked.connect(self.apply_config)
        button_layout.addWidget(self.apply_button)
        
        layout.addLayout(button_layout)
        
    def create_model_selection_tab(self) -> QWidget:
        """Crea la pestaña de selección de modelo"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tipo de modelo
        model_group = QGroupBox("Tipo de Modelo")
        model_layout = QFormLayout(model_group)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "BallisticCNN - Clasificación de características",
            "SiameseNetwork - Comparación de similitud",
            "TripletNetwork - Aprendizaje métrico",
            "HierarchicalClassifier - Clasificación jerárquica"
        ])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addRow("Tipo de Modelo:", self.model_type_combo)
        
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems([
            "custom - Arquitectura personalizada",
            "resnet18 - ResNet-18 preentrenado",
            "resnet50 - ResNet-50 preentrenado",
            "efficientnet_b0 - EfficientNet-B0",
            "efficientnet_b3 - EfficientNet-B3"
        ])
        model_layout.addRow("Arquitectura Base:", self.architecture_combo)
        
        layout.addWidget(model_group)
        
        # Parámetros del modelo
        params_group = QGroupBox("Parámetros del Modelo")
        params_layout = QFormLayout(params_group)
        
        # Tamaño de entrada
        input_size_layout = QHBoxLayout()
        self.input_width_spin = QSpinBox()
        self.input_width_spin.setRange(64, 1024)
        self.input_width_spin.setValue(224)
        self.input_height_spin = QSpinBox()
        self.input_height_spin.setRange(64, 1024)
        self.input_height_spin.setValue(224)
        input_size_layout.addWidget(self.input_width_spin)
        input_size_layout.addWidget(QLabel("x"))
        input_size_layout.addWidget(self.input_height_spin)
        params_layout.addRow("Tamaño de Entrada:", input_size_layout)
        
        # Dimensión de características
        self.feature_dim_spin = QSpinBox()
        self.feature_dim_spin.setRange(64, 2048)
        self.feature_dim_spin.setValue(512)
        params_layout.addRow("Dim. Características:", self.feature_dim_spin)
        
        # Dropout
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.1)
        self.dropout_spin.setValue(0.3)
        params_layout.addRow("Dropout Rate:", self.dropout_spin)
        
        layout.addWidget(params_group)
        
        # Configuración específica
        self.specific_group = QGroupBox("Configuración Específica")
        self.specific_layout = QFormLayout(self.specific_group)
        
        # Atención
        self.use_attention_cb = QCheckBox("Usar mecanismo de atención")
        self.use_attention_cb.setChecked(True)
        self.specific_layout.addRow("", self.use_attention_cb)
        
        self.attention_type_combo = QComboBox()
        self.attention_type_combo.addItems(["spatial", "channel", "both"])
        self.specific_layout.addRow("Tipo de Atención:", self.attention_type_combo)
        
        layout.addWidget(self.specific_group)
        
        layout.addStretch()
        return tab
        
    def create_data_config_tab(self) -> QWidget:
        """Crea la pestaña de configuración de datos"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Configuración de dataset
        dataset_group = QGroupBox("Configuración del Dataset")
        dataset_layout = QFormLayout(dataset_group)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 256)
        self.batch_size_spin.setValue(32)
        dataset_layout.addRow("Batch Size:", self.batch_size_spin)
        
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 16)
        self.num_workers_spin.setValue(4)
        dataset_layout.addRow("Num Workers:", self.num_workers_spin)
        
        layout.addWidget(dataset_group)
        
        # Splits de datos
        splits_group = QGroupBox("División de Datos")
        splits_layout = QFormLayout(splits_group)
        
        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.1, 0.9)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.7)
        splits_layout.addRow("Entrenamiento:", self.train_split_spin)
        
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.15)
        splits_layout.addRow("Validación:", self.val_split_spin)
        
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.05, 0.5)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.15)
        splits_layout.addRow("Prueba:", self.test_split_spin)
        
        layout.addWidget(splits_group)
        
        # Augmentación
        aug_group = QGroupBox("Augmentación de Datos")
        aug_layout = QFormLayout(aug_group)
        
        self.use_augmentation_cb = QCheckBox("Habilitar augmentación")
        self.use_augmentation_cb.setChecked(True)
        aug_layout.addRow("", self.use_augmentation_cb)
        
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(0.0, 45.0)
        self.rotation_spin.setValue(15.0)
        aug_layout.addRow("Rotación (grados):", self.rotation_spin)
        
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.1)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.01)
        aug_layout.addRow("Ruido (std):", self.noise_spin)
        
        layout.addWidget(aug_group)
        
        layout.addStretch()
        return tab
        
    def create_training_config_tab(self) -> QWidget:
        """Crea la pestaña de configuración de entrenamiento"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Parámetros básicos
        basic_group = QGroupBox("Parámetros Básicos")
        basic_layout = QFormLayout(basic_group)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        basic_layout.addRow("Épocas:", self.epochs_spin)
        
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(0.001)
        basic_layout.addRow("Learning Rate:", self.learning_rate_spin)
        
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1e-2)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(1e-4)
        basic_layout.addRow("Weight Decay:", self.weight_decay_spin)
        
        layout.addWidget(basic_group)
        
        # Optimizador
        optimizer_group = QGroupBox("Optimizador")
        optimizer_layout = QFormLayout(optimizer_group)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw"])
        optimizer_layout.addRow("Tipo:", self.optimizer_combo)
        
        layout.addWidget(optimizer_group)
        
        # Scheduler
        scheduler_group = QGroupBox("Learning Rate Scheduler")
        scheduler_layout = QFormLayout(scheduler_group)
        
        self.use_scheduler_cb = QCheckBox("Usar scheduler")
        self.use_scheduler_cb.setChecked(True)
        scheduler_layout.addRow("", self.use_scheduler_cb)
        
        self.scheduler_type_combo = QComboBox()
        self.scheduler_type_combo.addItems(["cosine", "step", "plateau"])
        scheduler_layout.addRow("Tipo:", self.scheduler_type_combo)
        
        layout.addWidget(scheduler_group)
        
        # Early Stopping
        early_stop_group = QGroupBox("Early Stopping")
        early_stop_layout = QFormLayout(early_stop_group)
        
        self.use_early_stopping_cb = QCheckBox("Usar early stopping")
        self.use_early_stopping_cb.setChecked(True)
        early_stop_layout.addRow("", self.use_early_stopping_cb)
        
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(15)
        early_stop_layout.addRow("Paciencia:", self.patience_spin)
        
        layout.addWidget(early_stop_group)
        
        layout.addStretch()
        return tab
        
    def create_preview_tab(self) -> QWidget:
        """Crea la pestaña de vista previa"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        preview_label = QLabel("Vista Previa de la Configuración")
        preview_label.setProperty("class", "subtitle")
        layout.addWidget(preview_label)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.preview_text)
        
        # Botón para actualizar vista previa
        update_button = QPushButton("🔄 Actualizar Vista Previa")
        update_button.clicked.connect(self.update_preview)
        layout.addWidget(update_button)
        
        return tab
        
    def on_model_type_changed(self, model_text: str):
        """Maneja el cambio de tipo de modelo"""
        model_type = model_text.split(' - ')[0]
        
        # Limpiar configuración específica anterior
        for i in reversed(range(self.specific_layout.count())):
            child = self.specific_layout.itemAt(i)
            if child.widget():
                child.widget().setParent(None)
        
        # Configuración específica según el tipo de modelo
        if model_type == "SiameseNetwork":
            self.similarity_metric_combo = QComboBox()
            self.similarity_metric_combo.addItems(["cosine", "euclidean", "learned"])
            self.specific_layout.addRow("Métrica de Similitud:", self.similarity_metric_combo)
            
            self.margin_spin = QDoubleSpinBox()
            self.margin_spin.setRange(0.1, 5.0)
            self.margin_spin.setValue(1.0)
            self.specific_layout.addRow("Margen:", self.margin_spin)
            
        elif model_type == "BallisticCNN":
            self.num_classes_spin = QSpinBox()
            self.num_classes_spin.setRange(2, 1000)
            self.num_classes_spin.setValue(10)
            self.specific_layout.addRow("Número de Clases:", self.num_classes_spin)
            
    def load_current_config(self):
        """Carga la configuración actual si existe"""
        if not self.current_config:
            return
            
        # Cargar configuración del modelo
        if 'model_type' in self.current_config:
            model_type = self.current_config['model_type']
            for i in range(self.model_type_combo.count()):
                if self.model_type_combo.itemText(i).startswith(model_type):
                    self.model_type_combo.setCurrentIndex(i)
                    break
                    
        # Cargar otros parámetros...
        
    def get_current_config(self) -> Dict[str, Any]:
        """Obtiene la configuración actual del diálogo"""
        config = {
            'model_type': self.model_type_combo.currentText().split(' - ')[0],
            'architecture': self.architecture_combo.currentText().split(' - ')[0],
            'input_size': (self.input_width_spin.value(), self.input_height_spin.value()),
            'feature_dim': self.feature_dim_spin.value(),
            'dropout_rate': self.dropout_spin.value(),
            'use_attention': self.use_attention_cb.isChecked(),
            'attention_type': self.attention_type_combo.currentText(),
            'batch_size': self.batch_size_spin.value(),
            'num_workers': self.num_workers_spin.value(),
            'train_split': self.train_split_spin.value(),
            'val_split': self.val_split_spin.value(),
            'test_split': self.test_split_spin.value(),
            'use_augmentation': self.use_augmentation_cb.isChecked(),
            'rotation_range': self.rotation_spin.value(),
            'noise_std': self.noise_spin.value(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'use_scheduler': self.use_scheduler_cb.isChecked(),
            'scheduler_type': self.scheduler_type_combo.currentText(),
            'use_early_stopping': self.use_early_stopping_cb.isChecked(),
            'patience': self.patience_spin.value()
        }
        
        # Agregar configuración específica del modelo
        model_type = config['model_type']
        if model_type == "SiameseNetwork" and hasattr(self, 'similarity_metric_combo'):
            config['similarity_metric'] = self.similarity_metric_combo.currentText()
            config['margin'] = self.margin_spin.value()
        elif model_type == "BallisticCNN" and hasattr(self, 'num_classes_spin'):
            config['num_classes'] = self.num_classes_spin.value()
            
        return config
        
    def update_preview(self):
        """Actualiza la vista previa de la configuración"""
        config = self.get_current_config()
        
        preview_text = "=== CONFIGURACIÓN DE DEEP LEARNING ===\n\n"
        
        preview_text += "MODELO:\n"
        preview_text += f"  Tipo: {config['model_type']}\n"
        preview_text += f"  Arquitectura: {config['architecture']}\n"
        preview_text += f"  Tamaño de entrada: {config['input_size']}\n"
        preview_text += f"  Dimensión de características: {config['feature_dim']}\n"
        preview_text += f"  Dropout: {config['dropout_rate']}\n\n"
        
        preview_text += "DATOS:\n"
        preview_text += f"  Batch size: {config['batch_size']}\n"
        preview_text += f"  Workers: {config['num_workers']}\n"
        preview_text += f"  Split train/val/test: {config['train_split']:.2f}/{config['val_split']:.2f}/{config['test_split']:.2f}\n"
        preview_text += f"  Augmentación: {'Sí' if config['use_augmentation'] else 'No'}\n\n"
        
        preview_text += "ENTRENAMIENTO:\n"
        preview_text += f"  Épocas: {config['epochs']}\n"
        preview_text += f"  Learning rate: {config['learning_rate']:.6f}\n"
        preview_text += f"  Weight decay: {config['weight_decay']:.6f}\n"
        preview_text += f"  Optimizador: {config['optimizer']}\n"
        preview_text += f"  Scheduler: {'Sí' if config['use_scheduler'] else 'No'} ({config['scheduler_type']})\n"
        preview_text += f"  Early stopping: {'Sí' if config['use_early_stopping'] else 'No'} (paciencia: {config['patience']})\n"
        
        self.preview_text.setPlainText(preview_text)
        
    def reset_config(self):
        """Restablece la configuración a valores por defecto"""
        reply = QMessageBox.question(
            self, "Confirmar", 
            "¿Está seguro de que desea restablecer toda la configuración?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Restablecer todos los controles a valores por defecto
            self.model_type_combo.setCurrentIndex(0)
            self.architecture_combo.setCurrentIndex(0)
            self.input_width_spin.setValue(224)
            self.input_height_spin.setValue(224)
            self.feature_dim_spin.setValue(512)
            self.dropout_spin.setValue(0.3)
            # ... restablecer otros controles
            
    def apply_config(self):
        """Aplica la configuración y cierra el diálogo"""
        config = self.get_current_config()
        
        # Validar configuración
        if abs(config['train_split'] + config['val_split'] + config['test_split'] - 1.0) > 1e-6:
            QMessageBox.warning(
                self, "Error de Configuración",
                "Los splits de datos deben sumar 1.0"
            )
            return
            
        self.modelConfigured.emit(config)
        self.accept()