import os
import sys
from pathlib import Path
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication

# Añadir raíz del proyecto al sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.widgets.analysis.configuration_levels_manager import ConfigurationLevelsManager


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_configuration_levels_manager_reset_alias_and_levels():
    app = get_app()

    cm = ConfigurationLevelsManager()

    # Nivel inicial
    assert cm.get_current_level() == 'basic'

    # Cambiar nivel
    cm.set_level('advanced')
    assert cm.get_current_level() == 'advanced'

    # Alias reset_to_defaults debe devolver a 'basic'
    cm.reset_to_defaults()
    assert cm.get_current_level() == 'basic'


def test_configuration_levels_manager_validation_signal():
    app = get_app()

    cm = ConfigurationLevelsManager()
    captured = []
    cm.validationChanged.connect(lambda valid: captured.append(valid))

    # Activar configuración válida vía UI (no por dict local)
    cm.set_level('basic')
    cm.basic_nist.enable_nist_cb.setChecked(True)
    cm.basic_afte.enable_afte_cb.setChecked(False)

    # Forzar validación interna
    cm._validate_configuration()

    assert captured[-1] is True