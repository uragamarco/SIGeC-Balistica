"""
Configuration Levels Manager for SIGeC-Balisticar Analysis Tab.

This widget manages hierarchical configuration levels (Basic → Intermediate → Advanced)
providing progressive access to configuration options based on user needs.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QButtonGroup, QFrame, QScrollArea,
                            QStackedWidget, QGroupBox, QFormLayout)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon

from ..shared.nist_config_widget import NISTConfigurationWidget
from ..shared.afte_config_widget import AFTEConfigurationWidget
from ..shared.dl_config_widget import DeepLearningConfigWidget
from ..shared.image_processing_widget import ImageProcessingWidget


class ConfigurationLevelsManager(QWidget):
    """
    Manages configuration levels for ballistic analysis.
    
    Provides three levels of configuration:
    - Basic: Essential settings for quick analysis
    - Intermediate: Additional options for detailed analysis
    - Advanced: Full control over all parameters
    """
    
    # Signals
    levelChanged = pyqtSignal(str)  # 'basic', 'intermediate', 'advanced'
    configurationChanged = pyqtSignal(dict)
    validationChanged = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current configuration level
        self.current_level = 'basic'
        
        # Configuration data
        self.configuration = {
            'basic': {},
            'intermediate': {},
            'advanced': {}
        }
        
        # Setup UI
        self._setup_ui()
        self._setup_connections()
        
        # Initialize with basic level
        self.set_level('basic')
    
    def _setup_ui(self):
        """Setup the configuration levels UI."""
        self.setObjectName("configuration-levels-manager")
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Level selector header
        self._create_level_selector()
        layout.addWidget(self.level_selector_frame)
        
        # Configuration content area
        self._create_configuration_area()
        layout.addWidget(self.config_scroll_area)
        
        # Quick actions footer
        self._create_quick_actions()
        layout.addWidget(self.quick_actions_frame)
    
    def _create_level_selector(self):
        """Create the level selector buttons."""
        self.level_selector_frame = QFrame()
        self.level_selector_frame.setObjectName("level-selector")
        self.level_selector_frame.setStyleSheet("""
            #level-selector {
                background-color: #ffffff;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(self.level_selector_frame)
        
        # Title
        title = QLabel("Nivel de Configuración")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #343a40; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        self.level_description = QLabel("Selecciona el nivel de configuración según tu experiencia y necesidades.")
        self.level_description.setStyleSheet("color: #6c757d; margin-bottom: 15px;")
        self.level_description.setWordWrap(True)
        layout.addWidget(self.level_description)
        
        # Level buttons
        buttons_layout = QHBoxLayout()
        self.level_button_group = QButtonGroup()
        
        # Basic level button
        self.basic_button = self._create_level_button(
            "Básico", 
            "Configuración esencial para análisis rápido",
            "#28a745"
        )
        self.basic_button.setChecked(True)
        
        # Intermediate level button
        self.intermediate_button = self._create_level_button(
            "Intermedio",
            "Opciones adicionales para análisis detallado", 
            "#ffc107"
        )
        
        # Advanced level button
        self.advanced_button = self._create_level_button(
            "Avanzado",
            "Control completo sobre todos los parámetros",
            "#dc3545"
        )
        
        self.level_button_group.addButton(self.basic_button, 0)
        self.level_button_group.addButton(self.intermediate_button, 1)
        self.level_button_group.addButton(self.advanced_button, 2)
        
        buttons_layout.addWidget(self.basic_button)
        buttons_layout.addWidget(self.intermediate_button)
        buttons_layout.addWidget(self.advanced_button)
        
        layout.addLayout(buttons_layout)
    
    def _create_level_button(self, title, description, color):
        """Create a level selection button."""
        button = QPushButton()
        button.setCheckable(True)
        button.setMinimumHeight(80)
        button.setStyleSheet(f"""
            QPushButton {{
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: #ffffff;
                text-align: left;
                padding: 10px 15px;
                font-weight: 600;
                color: #495057;
            }}
            QPushButton:checked {{
                border-color: {color};
                background-color: {color}15;
                color: {color};
            }}
            QPushButton:hover {{
                border-color: {color};
                background-color: {color}08;
            }}
        """)
        
        # Create button content
        content_layout = QVBoxLayout()
        content_layout.setSpacing(5)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        
        desc_label = QLabel(description)
        desc_label.setFont(QFont("Segoe UI", 8))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #6c757d;")
        
        content_layout.addWidget(title_label)
        content_layout.addWidget(desc_label)
        
        button.setLayout(content_layout)
        return button
    
    def _create_configuration_area(self):
        """Create the scrollable configuration area."""
        self.config_scroll_area = QScrollArea()
        self.config_scroll_area.setWidgetResizable(True)
        self.config_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.config_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        
        # Stacked widget for different levels
        self.config_stack = QStackedWidget()
        
        # Create configuration widgets for each level
        self._create_basic_config()
        self._create_intermediate_config()
        self._create_advanced_config()
        
        self.config_scroll_area.setWidget(self.config_stack)
    
    def _create_basic_config(self):
        """Create basic level configuration."""
        self.basic_config = QWidget()
        layout = QVBoxLayout(self.basic_config)
        layout.setSpacing(20)
        
        # Basic configuration title
        title = QLabel("Configuración Básica")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #28a745; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Configuración esencial para realizar un análisis balístico básico. "
                     "Estas opciones cubren los casos de uso más comunes.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6c757d; margin-bottom: 15px;")
        layout.addWidget(desc)
        
        # Quick configuration options
        quick_config = QGroupBox("Configuración Rápida")
        quick_layout = QFormLayout(quick_config)
        
        # Basic NIST options (simplified)
        self.basic_nist = NISTConfigurationWidget()
        self.basic_nist.set_mode('basic')
        layout.addWidget(self.basic_nist)
        
        # Basic AFTE options (simplified)
        self.basic_afte = AFTEConfigurationWidget()
        self.basic_afte.set_mode('basic')
        layout.addWidget(self.basic_afte)
        
        layout.addStretch()
        self.config_stack.addWidget(self.basic_config)
    
    def _create_intermediate_config(self):
        """Create intermediate level configuration."""
        self.intermediate_config = QWidget()
        layout = QVBoxLayout(self.intermediate_config)
        layout.setSpacing(20)
        
        # Intermediate configuration title
        title = QLabel("Configuración Intermedia")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #ffc107; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Configuración detallada con opciones adicionales para análisis más precisos. "
                     "Incluye configuraciones de Deep Learning y procesamiento de imágenes.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6c757d; margin-bottom: 15px;")
        layout.addWidget(desc)
        
        # NIST configuration (intermediate)
        self.intermediate_nist = NISTConfigurationWidget()
        self.intermediate_nist.set_mode('intermediate')
        layout.addWidget(self.intermediate_nist)
        
        # AFTE configuration (intermediate)
        self.intermediate_afte = AFTEConfigurationWidget()
        self.intermediate_afte.set_mode('intermediate')
        layout.addWidget(self.intermediate_afte)
        
        # Deep Learning configuration (basic)
        self.intermediate_dl = DeepLearningConfigWidget()
        self.intermediate_dl.set_mode('basic')
        layout.addWidget(self.intermediate_dl)
        
        layout.addStretch()
        self.config_stack.addWidget(self.intermediate_config)
    
    def _create_advanced_config(self):
        """Create advanced level configuration."""
        self.advanced_config = QWidget()
        layout = QVBoxLayout(self.advanced_config)
        layout.setSpacing(20)
        
        # Advanced configuration title
        title = QLabel("Configuración Avanzada")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet("color: #dc3545; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Configuración completa con acceso a todos los parámetros disponibles. "
                     "Recomendado para usuarios expertos que requieren control total sobre el análisis.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #6c757d; margin-bottom: 15px;")
        layout.addWidget(desc)
        
        # Full NIST configuration
        self.advanced_nist = NISTConfigurationWidget()
        self.advanced_nist.set_mode('advanced')
        layout.addWidget(self.advanced_nist)
        
        # Full AFTE configuration
        self.advanced_afte = AFTEConfigurationWidget()
        self.advanced_afte.set_mode('advanced')
        layout.addWidget(self.advanced_afte)
        
        # Full Deep Learning configuration
        self.advanced_dl = DeepLearningConfigWidget()
        self.advanced_dl.set_mode('advanced')
        layout.addWidget(self.advanced_dl)
        
        # Image Processing configuration
        self.advanced_img = ImageProcessingWidget()
        layout.addWidget(self.advanced_img)
        
        layout.addStretch()
        self.config_stack.addWidget(self.advanced_config)
    
    def _create_quick_actions(self):
        """Create quick actions footer."""
        self.quick_actions_frame = QFrame()
        self.quick_actions_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QHBoxLayout(self.quick_actions_frame)
        
        # Preset configurations
        presets_label = QLabel("Configuraciones Predefinidas:")
        presets_label.setStyleSheet("font-weight: 600; color: #495057;")
        layout.addWidget(presets_label)
        
        # Quick preset buttons
        self.forensic_preset_btn = QPushButton("Forense Estándar")
        self.forensic_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        
        self.research_preset_btn = QPushButton("Investigación")
        self.research_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1e7e34;
            }
        """)
        
        self.custom_preset_btn = QPushButton("Personalizada")
        self.custom_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #545b62;
            }
        """)
        
        layout.addWidget(self.forensic_preset_btn)
        layout.addWidget(self.research_preset_btn)
        layout.addWidget(self.custom_preset_btn)
        
        layout.addStretch()
        
        # Validation status
        self.validation_label = QLabel("✓ Configuración válida")
        self.validation_label.setStyleSheet("color: #28a745; font-weight: 500;")
        layout.addWidget(self.validation_label)
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Level button connections
        self.level_button_group.buttonClicked.connect(self._on_level_changed)
        
        # Preset button connections
        self.forensic_preset_btn.clicked.connect(lambda: self.load_preset('forensic'))
        self.research_preset_btn.clicked.connect(lambda: self.load_preset('research'))
        self.custom_preset_btn.clicked.connect(lambda: self.load_preset('custom'))
    
    def _on_level_changed(self, button):
        """Handle level change."""
        level_map = {
            self.basic_button: 'basic',
            self.intermediate_button: 'intermediate',
            self.advanced_button: 'advanced'
        }
        
        new_level = level_map.get(button)
        if new_level:
            self.set_level(new_level)
    
    def set_level(self, level):
        """Set the current configuration level."""
        if level not in ['basic', 'intermediate', 'advanced']:
            return
        
        self.current_level = level
        
        # Update UI
        level_index = {'basic': 0, 'intermediate': 1, 'advanced': 2}[level]
        self.config_stack.setCurrentIndex(level_index)
        
        # Update level description
        descriptions = {
            'basic': "Configuración esencial para análisis rápido y eficiente.",
            'intermediate': "Opciones adicionales para análisis más detallado y preciso.",
            'advanced': "Control completo sobre todos los parámetros disponibles."
        }
        self.level_description.setText(descriptions[level])
        
        # Emit signal
        self.levelChanged.emit(level)
        
        # Validate configuration
        self._validate_configuration()
    
    def get_configuration(self):
        """Get the current configuration based on the active level."""
        config = {}
        
        if self.current_level == 'basic':
            config['nist'] = self.basic_nist.get_configuration()
            config['afte'] = self.basic_afte.get_configuration()
        elif self.current_level == 'intermediate':
            config['nist'] = self.intermediate_nist.get_configuration()
            config['afte'] = self.intermediate_afte.get_configuration()
            config['deep_learning'] = self.intermediate_dl.get_configuration()
        elif self.current_level == 'advanced':
            config['nist'] = self.advanced_nist.get_configuration()
            config['afte'] = self.advanced_afte.get_configuration()
            config['deep_learning'] = self.advanced_dl.get_configuration()
            config['image_processing'] = self.advanced_img.get_configuration()
        
        return config
    
    def set_configuration(self, config):
        """Set configuration values."""
        # Apply configuration to appropriate widgets based on current level
        if self.current_level == 'basic':
            if 'nist' in config:
                self.basic_nist.set_configuration(config['nist'])
            if 'afte' in config:
                self.basic_afte.set_configuration(config['afte'])
        elif self.current_level == 'intermediate':
            if 'nist' in config:
                self.intermediate_nist.set_configuration(config['nist'])
            if 'afte' in config:
                self.intermediate_afte.set_configuration(config['afte'])
            if 'deep_learning' in config:
                self.intermediate_dl.set_configuration(config['deep_learning'])
        elif self.current_level == 'advanced':
            if 'nist' in config:
                self.advanced_nist.set_configuration(config['nist'])
            if 'afte' in config:
                self.advanced_afte.set_configuration(config['afte'])
            if 'deep_learning' in config:
                self.advanced_dl.set_configuration(config['deep_learning'])
            if 'image_processing' in config:
                self.advanced_img.set_configuration(config['image_processing'])
        
        self._validate_configuration()
    
    def load_preset(self, preset_name):
        """Load a predefined configuration preset."""
        presets = {
            'forensic': {
                'nist': {'enabled': True, 'validation_level': 'standard'},
                'afte': {'enabled': True, 'method': 'cmc', 'threshold': 0.8}
            },
            'research': {
                'nist': {'enabled': False},
                'afte': {'enabled': True, 'method': 'advanced', 'threshold': 0.9},
                'deep_learning': {'enabled': True, 'confidence_threshold': 0.85}
            },
            'custom': {}
        }
        
        preset_config = presets.get(preset_name, {})
        self.set_configuration(preset_config)
    
    def _validate_configuration(self):
        """Validate the current configuration."""
        config = self.get_configuration()
        is_valid = True
        
        # Basic validation logic
        if self.current_level in ['basic', 'intermediate', 'advanced']:
            # Check if at least one analysis method is enabled
            nist_enabled = config.get('nist', {}).get('enabled', False)
            afte_enabled = config.get('afte', {}).get('enabled', False)
            dl_enabled = config.get('deep_learning', {}).get('enabled', False)
            
            is_valid = nist_enabled or afte_enabled or dl_enabled
        
        # Update validation UI
        if is_valid:
            self.validation_label.setText("✓ Configuración válida")
            self.validation_label.setStyleSheet("color: #28a745; font-weight: 500;")
        else:
            self.validation_label.setText("⚠ Configuración incompleta")
            self.validation_label.setStyleSheet("color: #dc3545; font-weight: 500;")
        
        # Emit validation signal
        self.validationChanged.emit(is_valid)
        
        return is_valid
    
    def get_current_level(self):
        """Get the current configuration level."""
        return self.current_level
    
    def reset_configuration(self):
        """Reset configuration to defaults."""
        # Reset all widgets to default state (fallback to set_configuration({}) if reset not available)
        def safe_reset(widget):
            if widget is None:
                return
            if hasattr(widget, 'reset_configuration') and callable(getattr(widget, 'reset_configuration')):
                try:
                    widget.reset_configuration()
                    return
                except Exception:
                    pass
            if hasattr(widget, 'set_configuration') and callable(getattr(widget, 'set_configuration')):
                try:
                    widget.set_configuration({})
                except Exception:
                    pass
        
        safe_reset(getattr(self, 'basic_nist', None))
        safe_reset(getattr(self, 'basic_afte', None))
        safe_reset(getattr(self, 'intermediate_nist', None))
        safe_reset(getattr(self, 'intermediate_afte', None))
        safe_reset(getattr(self, 'intermediate_dl', None))
        safe_reset(getattr(self, 'advanced_nist', None))
        safe_reset(getattr(self, 'advanced_afte', None))
        safe_reset(getattr(self, 'advanced_dl', None))
        safe_reset(getattr(self, 'advanced_img', None))
        
        # Reset to basic level
        self.set_level('basic')
        self.basic_button.setChecked(True)

    def reset_to_defaults(self):
        """Compatibility alias to reset all configurations to default values."""
        self.reset_configuration()