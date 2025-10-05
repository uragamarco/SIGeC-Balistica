#!/usr/bin/env python3
"""
Módulo GUI para Sistema Balístico Forense SIGeC-Balistica
Interfaz de usuario moderna y profesional
"""

__version__ = "0.1.3"
__author__ = "SIGeC-BalisticaDevelopment Team"

# Importaciones principales
from .main_window import MainWindow
from .styles import SIGeCBallisticaTheme, apply_SIGeC_Balistica_theme
from .shared_widgets import (
    ImageDropZone, 
    ResultCard, 
    CollapsiblePanel,
    StepIndicator,
    ProgressCard
)

__all__ = [
    'MainWindow',
    'SIGeCBallisticaTheme', 
    'apply_SIGeC_Balistica_theme',
    'ImageDropZone',
    'ResultCard',
    'CollapsiblePanel', 
    'StepIndicator',
    'ProgressCard'
]