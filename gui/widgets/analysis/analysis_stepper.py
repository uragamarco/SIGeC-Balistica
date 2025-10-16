"""
Analysis Stepper Widget for SIGeC-Balisticar Analysis Tab.

This widget provides a step-by-step navigation specifically designed for
the ballistic analysis workflow, with optimized steps and validation.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont
from ..shared.stepper_widget import StepperWidget


class AnalysisStepper(StepperWidget):
    """
    Specialized stepper widget for ballistic analysis workflow.
    
    Provides a guided step-by-step process for:
    1. Image Selection and Validation
    2. Configuration Setup (Basic → Advanced)
    3. Analysis Execution
    4. Results Review and Export
    """
    
    # Analysis-specific signals
    imageSelectionRequested = pyqtSignal()
    configurationRequested = pyqtSignal(str)  # level: 'basic', 'intermediate', 'advanced'
    analysisExecutionRequested = pyqtSignal()
    resultsReviewRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    nextRequested = pyqtSignal()  # Signal for next step request
    previousRequested = pyqtSignal()  # Signal for previous step request
    
    def __init__(self, parent=None):
        # Define analysis-specific steps
        analysis_steps = [
            {
                'id': 'image_selection',
                'title': 'Selección de Imágenes',
                'description': 'Cargar y validar imágenes balísticas',
                'icon': '📁'
            },
            {
                'id': 'configuration',
                'title': 'Configuración',
                'description': 'Configurar parámetros de análisis',
                'icon': '⚙️'
            },
            {
                'id': 'analysis',
                'title': 'Análisis',
                'description': 'Ejecutar análisis balístico',
                'icon': '🔍'
            },
            {
                'id': 'results',
                'title': 'Resultados',
                'description': 'Revisar y exportar resultados',
                'icon': '📊'
            }
        ]
        
        super().__init__(analysis_steps, parent)
        
        # Analysis-specific state
        self.selected_images = []
        self.configuration_level = 'basic'
        self.analysis_results = None
        
        # Connect step-specific actions
        self.stepActivated.connect(self._handle_step_activation)
        
        # Setup analysis-specific UI
        self._setup_analysis_ui()
    
    def _setup_analysis_ui(self):
        """Setup analysis-specific UI elements."""
        # Add analysis status indicators
        self.status_layout = QHBoxLayout()
        
        # Image count indicator
        self.image_count_label = QLabel("Imágenes: 0")
        self.image_count_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px 10px;
                color: #6c757d;
                font-weight: 500;
            }
        """)
        
        # Configuration level indicator
        self.config_level_label = QLabel("Configuración: Básica")
        self.config_level_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px 10px;
                color: #6c757d;
                font-weight: 500;
            }
        """)
        
        # Analysis status indicator
        self.analysis_status_label = QLabel("Estado: Pendiente")
        self.analysis_status_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 5px 10px;
                color: #6c757d;
                font-weight: 500;
            }
        """)
        
        self.status_layout.addWidget(self.image_count_label)
        self.status_layout.addWidget(self.config_level_label)
        self.status_layout.addWidget(self.analysis_status_label)
        self.status_layout.addStretch()
        
        # Insert status layout after stepper
        self.layout().insertLayout(1, self.status_layout)
    
    def _handle_step_activation(self, step_id):
        """Handle activation of analysis-specific steps."""
        if step_id == 'image_selection':
            self.imageSelectionRequested.emit()
        elif step_id == 'configuration':
            self.configurationRequested.emit(self.configuration_level)
        elif step_id == 'analysis':
            self.analysisExecutionRequested.emit()
        elif step_id == 'results':
            self.resultsReviewRequested.emit()
    
    def set_images_count(self, count):
        """Update the number of selected images."""
        self.selected_images = list(range(count))  # Simplified representation
        self.image_count_label.setText(f"Imágenes: {count}")
        
        # Update step validation
        if count > 0:
            self.set_step_status('image_selection', 'completed')
            self.enable_step('configuration')
            self.image_count_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 4px;
                    padding: 5px 10px;
                    color: #155724;
                    font-weight: 500;
                }
            """)
        else:
            self.set_step_status('image_selection', 'pending')
            self.disable_step('configuration')
            self.image_count_label.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 5px 10px;
                    color: #6c757d;
                    font-weight: 500;
                }
            """)
    
    def set_configuration_level(self, level):
        """Update the configuration level."""
        self.configuration_level = level
        level_names = {
            'basic': 'Básica',
            'intermediate': 'Intermedia',
            'advanced': 'Avanzada'
        }
        
        self.config_level_label.setText(f"Configuración: {level_names.get(level, 'Básica')}")
        
        # Update step validation
        self.set_step_status('configuration', 'completed')
        self.enable_step('analysis')
        
        # Update styling based on level
        colors = {
            'basic': ('#d4edda', '#c3e6cb', '#155724'),
            'intermediate': ('#fff3cd', '#ffeaa7', '#856404'),
            'advanced': ('#f8d7da', '#f5c6cb', '#721c24')
        }
        
        bg_color, border_color, text_color = colors.get(level, colors['basic'])
        self.config_level_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 5px 10px;
                color: {text_color};
                font-weight: 500;
            }}
        """)
    
    def set_analysis_status(self, status):
        """Update the analysis execution status."""
        status_info = {
            'pending': ('Pendiente', '#f8f9fa', '#dee2e6', '#6c757d'),
            'running': ('Ejecutando...', '#fff3cd', '#ffeaa7', '#856404'),
            'completed': ('Completado', '#d4edda', '#c3e6cb', '#155724'),
            'error': ('Error', '#f8d7da', '#f5c6cb', '#721c24')
        }
        
        text, bg_color, border_color, text_color = status_info.get(status, status_info['pending'])
        self.analysis_status_label.setText(f"Estado: {text}")
        self.analysis_status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 5px 10px;
                color: {text_color};
                font-weight: 500;
            }}
        """)
        
        # Update step validation based on status
        if status == 'completed':
            self.set_step_status('analysis', 'completed')
            self.enable_step('results')
        elif status == 'running':
            self.set_step_status('analysis', 'active')
        elif status == 'error':
            self.set_step_status('analysis', 'pending')
    
    def set_results_available(self, available=True):
        """Update results availability status."""
        if available:
            self.set_step_status('results', 'completed')
        else:
            self.set_step_status('results', 'pending')
    
    def validate_current_step(self):
        """Validate the current step and enable/disable navigation."""
        current_step = self.get_current_step()
        
        if current_step == 'image_selection':
            return len(self.selected_images) > 0
        elif current_step == 'configuration':
            return self.configuration_level is not None
        elif current_step == 'analysis':
            return len(self.selected_images) > 0 and self.configuration_level is not None
        elif current_step == 'results':
            return self.analysis_results is not None
        
        return False
    
    def reset_analysis(self):
        """Reset the analysis workflow to initial state."""
        self.selected_images = []
        self.configuration_level = 'basic'
        self.analysis_results = None
        
        # Reset all steps
        self.set_step_status('image_selection', 'pending')
        self.set_step_status('configuration', 'pending')
        self.set_step_status('analysis', 'pending')
        self.set_step_status('results', 'pending')
        
        # Reset to first step
        self.set_current_step('image_selection')
        
        # Reset status indicators
        self.set_images_count(0)
        self.set_configuration_level('basic')
        self.set_analysis_status('pending')
        self.set_results_available(False)
    
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
    
    def set_step_status(self, step_id: str, status: str):
        """Set the status of a specific step
        
        Args:
            step_id: Step identifier ('image_selection', 'configuration', 'analysis', 'results')
            status: Status string ('pending', 'active', 'completed', 'error')
        """
        # Map step IDs to step indices
        step_mapping = {
            'image_selection': 0,
            'configuration': 1,
            'analysis': 2,
            'results': 3
        }
        
        if step_id not in step_mapping:
            print(f"Warning: Unknown step_id '{step_id}' in set_step_status")
            return
            
        step_index = step_mapping[step_id]
        
        # Update step status based on the status parameter
        if status == 'completed':
            if step_index < len(self.step_widgets):
                # Mark step as completed
                self.mark_step_completed(step_index)
        elif status == 'active':
            # Set as current step
            self.set_current_step(step_index)
        elif status == 'pending':
            # Reset step to pending state
            if step_index < len(self.step_widgets):
                self.step_widgets[step_index].set_completed(False)
                self.step_widgets[step_index].set_active(False)
        elif status == 'error':
            # Handle error state (could be implemented with visual indicators)
            print(f"Step {step_id} encountered an error")
        
        # Force UI update
        self.update_steps()

    def update_step_descriptions(self, step_descriptions=None):
        """Update step descriptions dynamically."""
        if step_descriptions is None:
            # Use default descriptions
            step_descriptions = {
                'image_selection': 'Cargar y validar imágenes balísticas',
                'configuration': 'Configurar parámetros de análisis',
                'analysis': 'Ejecutar análisis balístico',
                'results': 'Revisar y exportar resultados'
            }
        
        # Update step descriptions in the steps list
        for i, step in enumerate(self.steps):
            step_id = step.get('id', f'step_{i}')
            if step_id in step_descriptions:
                step['description'] = step_descriptions[step_id]
        
        # Update step indicators if they exist
        if hasattr(self, 'step_indicators'):
            for i, indicator in enumerate(self.step_indicators):
                step_id = self.steps[i].get('id', f'step_{i}')
                if step_id in step_descriptions:
                    indicator.description = step_descriptions[step_id]
                    indicator.update()

    def reset(self):
        """Reset the stepper to initial state."""
        # Reset to first step
        self.current_step = 0
        
        # Reset all step states
        if hasattr(self, 'step_indicators'):
            for indicator in self.step_indicators:
                indicator.set_completed(False)
                indicator.set_active(False)
                indicator.set_enabled(True)
        
        # Reset internal state
        self.selected_images = []
        self.configuration_level = 'basic'
        
        # Reset steps status
        for step in self.steps:
            step['status'] = 'pending'
        
        # Set first step as active
        self.set_current_step(0)
        
        # Update UI
        self.update_steps()

    def get_analysis_summary(self):
        """Get a summary of the current analysis configuration."""
        return {
            'images_count': len(self.selected_images),
            'configuration_level': self.configuration_level,
            'current_step': self.get_current_step(),
            'completed_steps': [step_id for step_id, step in self.steps.items() 
                              if step['status'] == 'completed']
        }