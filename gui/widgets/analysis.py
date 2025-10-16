"""
Analysis-specific widgets for the SIGeC-Balisticar application
Includes AnalysisStepper and ConfigurationLevelsManager
"""

from typing import Dict, List, Optional, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QGroupBox, QFrame,
    QScrollArea, QTabWidget, QSpinBox, QDoubleSpinBox, QSlider,
    QTextEdit, QLineEdit, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor

from .shared import StepperWidget, NISTConfigurationWidget, AFTEConfigurationWidget, DeepLearningConfigWidget, ImageProcessingWidget


class AnalysisStepper(StepperWidget):
    """
    Specialized stepper widget for ballistic analysis workflow
    """
    
    # Additional signals specific to analysis
    nextRequested = pyqtSignal()
    previousRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        # Define analysis-specific steps
        steps = [
            "Image Selection",
            "Configuration", 
            "Analysis",
            "Results"
        ]
        
        super().__init__(steps, parent)
        
        # Analysis-specific properties
        self.analysis_status = "Pending"
        self.image_count = 0
        self.config_level = "Basic"
        
        # Add navigation buttons
        self.setup_navigation_buttons()
        
    def setup_navigation_buttons(self):
        """Add navigation buttons to the stepper"""
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setObjectName("stepperNavButton")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.previousRequested.emit)
        nav_layout.addWidget(self.prev_btn)
        
        nav_layout.addStretch()
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setObjectName("stepperNavButton")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.nextRequested.emit)
        nav_layout.addWidget(self.next_btn)
        
        self.main_layout.addLayout(nav_layout)
        
    def set_current_step(self, step: int):
        """Override to update navigation buttons"""
        super().set_current_step(step)
        self.update_navigation_buttons()
        
    def update_navigation_buttons(self):
        """Update navigation button states"""
        self.prev_btn.setEnabled(self.current_step > 0)
        self.next_btn.setEnabled(self.current_step < len(self.steps) - 1)
        
    def update_status(self, image_count: int = None, config_level: str = None, analysis_status: str = None):
        """Update the status information displayed in the stepper"""
        if image_count is not None:
            self.image_count = image_count
        if config_level is not None:
            self.config_level = config_level
        if analysis_status is not None:
            self.analysis_status = analysis_status
            
        # Update step descriptions based on current status
        self.update_step_descriptions()
        
    def update_step_descriptions(self):
        """Update step descriptions with current status"""
        descriptions = [
            f"Images: {self.image_count} selected",
            f"Level: {self.config_level}",
            f"Status: {self.analysis_status}",
            "View results and reports"
        ]
        
        for i, (step_widget, desc) in enumerate(zip(self.step_widgets, descriptions)):
            if hasattr(step_widget, 'description_label'):
                step_widget.description_label.setText(desc)


class ConfigurationLevelsManager(QWidget):
    """
    Hierarchical configuration manager for different analysis levels
    """
    
    # Signals
    configurationChanged = pyqtSignal(dict)
    levelChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration state
        self.current_level = "Basic"
        self.configurations = {
            "Basic": {},
            "Intermediate": {},
            "Advanced": {}
        }
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Level selector
        level_group = QGroupBox("Analysis Level")
        level_group.setObjectName("configLevelGroup")
        level_layout = QVBoxLayout(level_group)
        
        self.level_combo = QComboBox()
        self.level_combo.setObjectName("levelCombo")
        self.level_combo.addItems(["Basic", "Intermediate", "Advanced"])
        level_layout.addWidget(self.level_combo)
        
        # Level description
        self.level_description = QLabel()
        self.level_description.setObjectName("levelDescription")
        self.level_description.setWordWrap(True)
        level_layout.addWidget(self.level_description)
        
        layout.addWidget(level_group)
        
        # Configuration tabs
        self.config_tabs = QTabWidget()
        self.config_tabs.setObjectName("configTabs")
        
        # Image Processing tab
        self.image_processing_widget = ImageProcessingWidget()
        self.config_tabs.addTab(self.image_processing_widget, "Image Processing")
        
        # NIST Standards tab
        self.nist_widget = NISTConfigurationWidget()
        self.config_tabs.addTab(self.nist_widget, "NIST Standards")
        
        # AFTE Analysis tab
        self.afte_widget = AFTEConfigurationWidget()
        self.config_tabs.addTab(self.afte_widget, "AFTE Analysis")
        
        # Deep Learning tab
        self.dl_widget = DeepLearningConfigWidget()
        self.config_tabs.addTab(self.dl_widget, "Deep Learning")
        
        layout.addWidget(self.config_tabs)
        
        # Configuration summary
        summary_group = QGroupBox("Configuration Summary")
        summary_group.setObjectName("configSummaryGroup")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setObjectName("configSummary")
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # Initialize with basic level
        self.update_level_description()
        self.update_configuration_summary()
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.level_combo.currentTextChanged.connect(self.on_level_changed)
        
        # Connect configuration widget signals
        self.image_processing_widget.configurationChanged.connect(self.on_config_changed)
        self.nist_widget.configurationChanged.connect(self.on_config_changed)
        self.afte_widget.configurationChanged.connect(self.on_config_changed)
        self.dl_widget.configurationChanged.connect(self.on_config_changed)
        
    def on_level_changed(self, level: str):
        """Handle analysis level changes"""
        self.current_level = level
        self.update_level_description()
        self.update_widget_visibility()
        self.update_configuration_summary()
        self.levelChanged.emit(level)
        
    def update_level_description(self):
        """Update the level description text"""
        descriptions = {
            "Basic": "Quick analysis with essential features. Suitable for routine examinations.",
            "Intermediate": "Comprehensive analysis with NIST compliance and detailed reporting.",
            "Advanced": "Full forensic analysis with deep learning, AFTE conclusions, and research-grade metrics."
        }
        
        self.level_description.setText(descriptions.get(self.current_level, ""))
        
    def update_widget_visibility(self):
        """Update widget visibility based on selected level"""
        # Basic level: Only image processing
        if self.current_level == "Basic":
            self.config_tabs.setTabEnabled(1, False)  # NIST
            self.config_tabs.setTabEnabled(2, False)  # AFTE
            self.config_tabs.setTabEnabled(3, False)  # Deep Learning
            
        # Intermediate level: Image processing + NIST
        elif self.current_level == "Intermediate":
            self.config_tabs.setTabEnabled(1, True)   # NIST
            self.config_tabs.setTabEnabled(2, False)  # AFTE
            self.config_tabs.setTabEnabled(3, False)  # Deep Learning
            
        # Advanced level: All features
        elif self.current_level == "Advanced":
            self.config_tabs.setTabEnabled(1, True)   # NIST
            self.config_tabs.setTabEnabled(2, True)   # AFTE
            self.config_tabs.setTabEnabled(3, True)   # Deep Learning
            
    def on_config_changed(self, config: Dict[str, Any]):
        """Handle configuration changes from child widgets"""
        # Store configuration for current level
        self.configurations[self.current_level] = self.get_current_configuration()
        
        # Update summary and emit signal
        self.update_configuration_summary()
        self.configurationChanged.emit(self.configurations[self.current_level])
        
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get the current complete configuration"""
        config = {
            "level": self.current_level.lower(),
            "image_processing": self.image_processing_widget.get_configuration(),
        }
        
        # Add level-specific configurations
        if self.current_level in ["Intermediate", "Advanced"]:
            config["nist"] = self.nist_widget.get_configuration()
            
        if self.current_level == "Advanced":
            config["afte"] = self.afte_widget.get_configuration()
            config["deep_learning"] = self.dl_widget.get_configuration()
            
        return config
        
    def update_configuration_summary(self):
        """Update the configuration summary display"""
        config = self.get_current_configuration()
        
        summary_parts = [
            f"Analysis Level: {self.current_level}",
            f"Image Processing: {len(config.get('image_processing', {}))} options configured"
        ]
        
        if "nist" in config:
            summary_parts.append(f"NIST Standards: {'Enabled' if config['nist'].get('enabled', False) else 'Disabled'}")
            
        if "afte" in config:
            summary_parts.append(f"AFTE Analysis: {'Enabled' if config['afte'].get('enabled', False) else 'Disabled'}")
            
        if "deep_learning" in config:
            summary_parts.append(f"Deep Learning: {'Enabled' if config['deep_learning'].get('enabled', False) else 'Disabled'}")
            
        self.summary_text.setPlainText("\n".join(summary_parts))
        
    def get_current_level(self) -> str:
        """Get the current analysis level"""
        return self.current_level
        
    def set_level(self, level: str):
        """Set the analysis level programmatically"""
        if level in ["Basic", "Intermediate", "Advanced"]:
            self.level_combo.setCurrentText(level)
            
    def reset_to_defaults(self):
        """Reset all configurations to default values"""
        self.level_combo.setCurrentText("Basic")
        
        # Reset all child widgets
        self.image_processing_widget.reset_to_defaults()
        self.nist_widget.reset_to_defaults()
        self.afte_widget.reset_to_defaults()
        self.dl_widget.reset_to_defaults()
        
        # Clear configurations
        self.configurations = {
            "Basic": {},
            "Intermediate": {},
            "Advanced": {}
        }
        
        self.update_configuration_summary()


class AnalysisProgressWidget(QWidget):
    """
    Widget for displaying detailed analysis progress
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_step = ""
        self.progress_value = 0
        self.step_results = {}
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Main progress bar
        self.main_progress = QProgressBar()
        self.main_progress.setObjectName("mainProgressBar")
        self.main_progress.setRange(0, 100)
        layout.addWidget(self.main_progress)
        
        # Current step label
        self.step_label = QLabel("Ready to start analysis")
        self.step_label.setObjectName("stepLabel")
        layout.addWidget(self.step_label)
        
        # Detailed progress area
        self.details_area = QScrollArea()
        self.details_area.setObjectName("progressDetails")
        self.details_area.setMaximumHeight(150)
        self.details_area.setWidgetResizable(True)
        
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout(self.details_widget)
        self.details_area.setWidget(self.details_widget)
        
        layout.addWidget(self.details_area)
        
    def update_progress(self, progress: int, message: str):
        """Update the main progress"""
        self.progress_value = progress
        self.current_step = message
        
        self.main_progress.setValue(progress)
        self.step_label.setText(message)
        
    def add_step_result(self, step_name: str, result: Dict[str, Any]):
        """Add a completed step result"""
        self.step_results[step_name] = result
        
        # Create result widget
        result_widget = QFrame()
        result_widget.setObjectName("stepResult")
        result_layout = QHBoxLayout(result_widget)
        
        # Step name
        name_label = QLabel(f"✓ {step_name}")
        name_label.setObjectName("stepResultName")
        result_layout.addWidget(name_label)
        
        result_layout.addStretch()
        
        # Result summary
        if "processing_time" in result:
            time_label = QLabel(f"{result['processing_time']:.2f}s")
            time_label.setObjectName("stepResultTime")
            result_layout.addWidget(time_label)
            
        self.details_layout.addWidget(result_widget)
        
        # Scroll to bottom
        self.details_area.verticalScrollBar().setValue(
            self.details_area.verticalScrollBar().maximum()
        )
        
    def clear_progress(self):
        """Clear all progress information"""
        self.main_progress.setValue(0)
        self.step_label.setText("Ready to start analysis")
        
        # Clear details
        for i in reversed(range(self.details_layout.count())):
            child = self.details_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
                
        self.step_results.clear()


class ConfigurationPresetManager(QWidget):
    """
    Widget for managing configuration presets
    """
    
    # Signals
    presetLoaded = pyqtSignal(dict)
    presetSaved = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.presets = {
            "Quick Analysis": {
                "level": "basic",
                "image_processing": {"enhance_contrast": True, "denoise": True},
                "description": "Fast analysis for routine examinations"
            },
            "Forensic Standard": {
                "level": "intermediate", 
                "image_processing": {"enhance_contrast": True, "denoise": True, "edge_enhancement": True},
                "nist": {"enabled": True, "generate_report": True},
                "description": "Standard forensic analysis with NIST compliance"
            },
            "Research Grade": {
                "level": "advanced",
                "image_processing": {"enhance_contrast": True, "denoise": True, "edge_enhancement": True},
                "nist": {"enabled": True, "generate_report": True},
                "afte": {"enabled": True, "generate_conclusions": True},
                "deep_learning": {"enabled": True, "model_type": "CNN"},
                "description": "Complete analysis with all features enabled"
            }
        }
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Preset selector
        selector_layout = QHBoxLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("presetCombo")
        self.preset_combo.addItem("Select preset...")
        self.preset_combo.addItems(list(self.presets.keys()))
        selector_layout.addWidget(self.preset_combo)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.setObjectName("presetLoadButton")
        self.load_btn.setEnabled(False)
        selector_layout.addWidget(self.load_btn)
        
        layout.addLayout(selector_layout)
        
        # Preset description
        self.description_label = QLabel()
        self.description_label.setObjectName("presetDescription")
        self.description_label.setWordWrap(True)
        self.description_label.hide()
        layout.addWidget(self.description_label)
        
        # Save new preset
        save_layout = QHBoxLayout()
        
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setObjectName("presetNameEdit")
        self.preset_name_edit.setPlaceholderText("Enter preset name...")
        save_layout.addWidget(self.preset_name_edit)
        
        self.save_btn = QPushButton("Save Current")
        self.save_btn.setObjectName("presetSaveButton")
        save_layout.addWidget(self.save_btn)
        
        layout.addLayout(save_layout)
        
    def setup_connections(self):
        """Setup signal-slot connections"""
        self.preset_combo.currentTextChanged.connect(self.on_preset_selected)
        self.load_btn.clicked.connect(self.load_selected_preset)
        self.save_btn.clicked.connect(self.save_current_preset)
        
    def on_preset_selected(self, preset_name: str):
        """Handle preset selection"""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            self.description_label.setText(preset.get("description", ""))
            self.description_label.show()
            self.load_btn.setEnabled(True)
        else:
            self.description_label.hide()
            self.load_btn.setEnabled(False)
            
    def load_selected_preset(self):
        """Load the selected preset"""
        preset_name = self.preset_combo.currentText()
        if preset_name in self.presets:
            preset_config = self.presets[preset_name].copy()
            preset_config.pop("description", None)  # Remove description from config
            self.presetLoaded.emit(preset_config)
            
    def save_current_preset(self):
        """Save the current configuration as a new preset"""
        preset_name = self.preset_name_edit.text().strip()
        if not preset_name:
            return
            
        # This would typically get the current configuration from the parent
        # For now, we'll emit a signal to request it
        self.presetSaved.emit(preset_name)
        
        # Clear the input
        self.preset_name_edit.clear()
        
    def add_preset(self, name: str, config: Dict[str, Any], description: str = ""):
        """Add a new preset"""
        self.presets[name] = config.copy()
        if description:
            self.presets[name]["description"] = description
            
        # Update combo box
        if name not in [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]:
            self.preset_combo.addItem(name)