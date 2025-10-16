"""
Analysis-specific widgets package for SIGeC-Balisticar GUI.

This package contains widgets specifically designed for the analysis tab,
including the stepper implementation and configuration level managers.
"""

from .analysis_stepper import AnalysisStepper
from .configuration_levels_manager import ConfigurationLevelsManager

__all__ = [
    'AnalysisStepper',
    'ConfigurationLevelsManager'
]