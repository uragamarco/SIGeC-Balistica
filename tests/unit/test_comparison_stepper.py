import os
import sys
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Añadir raíz del proyecto al sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication

from gui.widgets.comparison import ComparisonStepper


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_comparison_stepper_descriptions_alias_and_single_update():
    app = get_app()

    cs = ComparisonStepper()

    # Validar alias en bloque (usa defaults si None)
    cs.update_step_descriptions()
    # Debe actualizar description de 'comparison_mode'
    idx_map = {s.get('id'): i for i, s in enumerate(cs.steps)}
    mode_idx = idx_map['comparison_mode']
    assert cs.step_indicators[mode_idx].description != ""

    # Actualizar descripción singular y validar
    cs.update_step_description('configuration', 'Configure all parameters for comparison')
    conf_idx = idx_map['configuration']
    assert cs.step_indicators[conf_idx].description == 'Configure all parameters for comparison'


def test_comparison_stepper_navigation_and_ids():
    app = get_app()

    cs = ComparisonStepper()

    # ID inicial
    assert cs.get_current_step_id() == 'image_selection'

    # navegar
    cs.next_step()
    assert cs.get_current_step_id() == 'comparison_mode'
    cs.set_current_step('results')
    assert cs.get_current_step_id() == 'results'