"""
Shared widgets package
Contains reusable widgets for NIST, AFTE, Deep Learning, and other common configurations
"""

from .stepper_widget import StepperWidget
from .nist_config_widget import NISTConfigurationWidget
from .afte_config_widget import AFTEConfigurationWidget
from .dl_config_widget import DeepLearningConfigWidget
from .image_processing_widget import ImageProcessingWidget

# Import additional widgets from shared_widgets.py
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_widgets import (
        NISTStandardsWidget,
        AFTEAnalysisWidget, 
        DeepLearningWidget,
        ImageProcessingWidget as ImageProcessingAnalysisWidget,
        ImageSelector,
        VisualizationPanel,
        ResultsPanel
    )
except ImportError:
    # Fallback if not available
    NISTStandardsWidget = None
    AFTEAnalysisWidget = None
    DeepLearningWidget = None
    ImageProcessingAnalysisWidget = None
    ImageSelector = None
    VisualizationPanel = None
    ResultsPanel = None

__all__ = [
    'StepperWidget',
    'NISTConfigurationWidget', 
    'AFTEConfigurationWidget',
    'DeepLearningConfigWidget',
    'ImageProcessingWidget',
    'NISTStandardsWidget',
    'AFTEAnalysisWidget',
    'DeepLearningWidget', 
    'ImageProcessingAnalysisWidget',
    'ImageSelector',
    'VisualizationPanel',
    'ResultsPanel'
]