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

from gui.widgets.shared.stepper_widget import StepperWidget


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_stepper_get_current_step_id_and_activation_signal():
    app = get_app()

    steps = [
        {'id': 'image_selection', 'title': 'Image Selection', 'description': 'Select images'},
        {'id': 'configuration', 'title': 'Configuration', 'description': 'Configure parameters'},
        {'id': 'analysis', 'title': 'Analysis', 'description': 'Run analysis'},
        {'id': 'results', 'title': 'Results', 'description': 'Review results'},
    ]

    w = StepperWidget(steps)

    # Estado inicial
    assert w.get_current_step() == 0
    assert w.get_current_step_id() == 'image_selection'

    # Capturar señal de activación con ID
    emitted = []
    w.stepActivated.connect(lambda sid: emitted.append(sid))

    # Cambiar por ID
    w.set_current_step('configuration')
    assert w.get_current_step() == 1
    assert w.get_current_step_id() == 'configuration'
    assert emitted[-1] == 'configuration'

    # Cambiar por índice
    w.set_current_step(3)
    assert w.get_current_step() == 3
    assert w.get_current_step_id() == 'results'
    assert emitted[-1] == 'results'


def test_stepper_enable_disable_by_id_and_index():
    app = get_app()

    steps = [
        {'id': 'image_selection', 'title': 'Image Selection', 'description': 'Select images'},
        {'id': 'configuration', 'title': 'Configuration', 'description': 'Configure parameters'},
        {'id': 'analysis', 'title': 'Analysis', 'description': 'Run analysis'},
    ]

    w = StepperWidget(steps)

    # Deshabilitar por ID
    w.disable_step('analysis')
    assert w.step_indicators[2].is_enabled is False

    # Intentar habilitar un paso futuro: la regla secuencial lo mantiene deshabilitado
    w.enable_step(2)
    assert w.step_indicators[2].is_enabled is False

    # Al llegar al paso, debe estar habilitado
    w.set_current_step(2)
    assert w.step_indicators[2].is_enabled is True