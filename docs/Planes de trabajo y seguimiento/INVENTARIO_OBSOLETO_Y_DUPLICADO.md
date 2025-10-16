# Inventario de Archivos Obsoletos y Duplicados

Este documento lista archivos, rutas y referencias consideradas obsoletas, duplicadas o inconsistentes en el proyecto SIGeC-Balistica, con propuesta de acción controlada.

## Estado y Criterios
- Considerado obsoleto: no referenciado por el código actual, reemplazado por `config/unified_config.py` u otros módulos unificados.
- Considerado duplicado: funcionalidad equivalente a otra fuente canónica (`UnifiedConfig`, `UnifiedPipeline`, etc.).
- Considerado inconsistente: menciona archivos esperados que no existen o rutas erróneas.

## Configuración
- `config/layered_config_manager.py` (inexistente, import roto)
  - Tipo: inconsistente
  - Referencia: `config/config_manager.py` (pre-parche)
  - Acción: reemplazo ya aplicado; mantener vigilancia por nuevas referencias.

- `unified_config_consolidated.yaml` (no existe)
  - Tipo: inconsistente
  - Referencia: `tests/test_basic_integration.py`
  - Acción: crear si se requiere una exportación consolidada, o actualizar test para usar `config/unified_config.yaml`, `config/unified_config_testing.yaml` o `config/unified_config_production.yaml` según el entorno.

- `config.yaml` en raíz del proyecto
  - Tipo: legacy
  - Referencia: migrada por `config/unified_config.py` (`_migrate_legacy_configs`)
  - Acción: mantener mientras existan respaldos; tras validar migración, eliminar controladamente usando `config/config_consolidator.py::cleanup_legacy_configs(confirm=True)`.

- `tests/config.yaml`
  - Tipo: legacy
  - Referencia: migrada por `config/unified_config.py`
  - Acción: igual que `config.yaml` de raíz.

- `utils/config.py`
  - Tipo: deprecado (wrapper de compatibilidad)
  - Referencia: múltiples scripts; wrapper ya redirige a `config.unified_config`
  - Acción: mantener temporalmente por compatibilidad; plan de eliminación cuando no existan imports a `utils.config` (fase 2 del plan).

- `config/config_layers.yaml`
  - Tipo: duplicado/legacy
  - Referencia: usado por herramientas legacy de consolidación; el runtime actual usa `UnifiedConfig`
  - Acción: mantener solo como material de migración; retiro programado tras validar que ningún módulo lo carga en runtime.

- `production.json` en `config/`
  - Tipo: legacy
  - Referencia: `config/config_manager.py` conserva ruta legacy
  - Acción: no usado por `UnifiedConfig`; migrar contenido relevante a `unified_config_production.yaml` y eliminar tras verificación.

- Backups y temporales
  - Patrones: `config/backups/*`, `*~`, `*.bak`, `__pycache__/`, `.pytest_cache/`
  - Tipo: auxiliares
  - Acción: limpiar con script controlado; mantener políticas de retención definidas.

## Otros Módulos
- Referencias a gestores de capas (layered manager)
  - Tipo: inconsistente
  - Referencia: no se encontraron archivos reales; parches aplicados para eliminar uso.
  - Acción: continuar escaneo durante integración GUI y Optimización.

## Propuesta de Siguiente Paso
1. Actualizar `tests/test_basic_integration.py` para no exigir `unified_config_consolidated.yaml` o generar dicho archivo mediante `UnifiedConfig.export_config()` como parte del setup de tests.
2. Ejecutar `config/config_consolidator.py` en modo reporte para confirmar fuentes legacy y planificar `cleanup_legacy_configs(confirm=True)`.
3. Añadir `pythonpath = .` en `pytest.ini` si persisten import errors en CI.
4. Consolidación de tests completada: eliminar referencias a tests en raíz ya retirados.

## Consolidación de Tests

### Cambios Aplicados
- Eliminados duplicados en raíz: `test_real_images.py`, `test_system_validation.py`, `test_simple_images.py`, `test_gui_complete.py`.
- Movido `test_assets_real.py` a `tests/legacy/test_assets_real.py`.
- Retirados tests externos no referenciados: `gui/test_gui.py`, `scripts/test_dl_integration.py`.

### Referencias Actualizadas
- Validación de sistema: usar `tests/test_validation_system.py`.
- Imágenes reales: usar `tests/test_real_images.py`.
- GUI consolidada: `tests/integration/test_frontend_integration_consolidated.py`.

## Evidencias
1. Búsqueda confirma inexistencia de `layered_config_manager.py` y `parallel_config_optimized.py`.
2. `config/unified_config.py` contiene `get_unified_config`, `save_config`, `export_config`, y migración automática desde `config.yaml`.
3. `pytest.ini` centraliza configuración de testpaths y marcadores.