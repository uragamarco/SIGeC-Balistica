#!/usr/bin/env python3
"""
Módulo GUI para Sistema Balístico Forense SIGeC-Balistica
Interfaz de usuario moderna y profesional
"""

__version__ = "2.0.0"
__author__ = "SIGeC-BalisticaDevelopment Team"

# Importaciones principales
from .main_window import MainWindow
from .styles import SIGeC-BalisticaTheme, apply_SIGeC-Balistica_theme
from .shared_widgets import (
    ImageDropZone, 
    ResultCard, 
    CollapsiblePanel,
    StepIndicator,
    ProgressCard
)

__all__ = [
    'MainWindow',
    'SIGeC-BalisticaTheme', 
    'apply_SIGeC-Balistica_theme',
    'ImageDropZone',
    'ResultCard',
    'CollapsiblePanel', 
    'StepIndicator',
    'ProgressCard'
]