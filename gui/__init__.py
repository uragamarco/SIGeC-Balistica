#!/usr/bin/env python3
"""
Módulo GUI para Sistema Balístico Forense SEACABAr
Interfaz de usuario moderna y profesional
"""

__version__ = "2.0.0"
__author__ = "SEACABAr Development Team"

# Importaciones principales
from .main_window import MainWindow
from .styles import SEACABArTheme, apply_seacaba_theme
from .shared_widgets import (
    ImageDropZone, 
    ResultCard, 
    CollapsiblePanel,
    StepIndicator,
    ProgressCard
)

__all__ = [
    'MainWindow',
    'SEACABArTheme', 
    'apply_seacaba_theme',
    'ImageDropZone',
    'ResultCard',
    'CollapsiblePanel', 
    'StepIndicator',
    'ProgressCard'
]