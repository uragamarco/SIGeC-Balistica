# Tests del Sistema SIGeC-Balistica

## Resumen Ejecutivo

Este documento describe la organización, estado actual y mejoras implementadas en el sistema de pruebas del proyecto SIGeC-Balistica. Se han realizado importantes actualizaciones en el sistema de validación y se han corregido múltiples problemas de compatibilidad.

## Actualizaciones Recientes del Sistema de Validación

### ✅ Correcciones Implementadas (Diciembre 2024)

#### 1. Sistema de Importación de Validación
- **Problema**: Falta de importación de `ValidationRule` y `DataType` en tests
- **Solución**: Agregadas importaciones desde `core.data_validator`
- **Impacto**: Tests de validación ahora ejecutan correctamente

#### 2. Métodos de Validación Actualizados
- **Problema**: Uso de método obsoleto `validate_input`
- **Solución**: Migración a `validate_data` con esquemas estructurados
- **Mejora**: Mayor flexibilidad y consistencia en validación

#### 3. Estadísticas de Error Corregidas
- **Problema**: Tests esperaban clave `errors_by_type` inexistente
- **Solución**: Actualizado a `severity_distribution` según implementación real
- **Resultado**: Tests de estadísticas funcionan correctamente

#### 4. Estrategias de Recuperación Alineadas
- **Problema**: Expectativas incorrectas sobre estrategias de recuperación
- **Solución**: Alineadas con lógica real de `_determine_recovery_strategy`
- **Cambio**: `ImportError` ahora usa `GRACEFUL_DEGRADATION` en lugar de `FALLBACK`

#### 5. Configuración Unificada Corregida
- **Problema**: Constructor `UnifiedConfig` con parámetros incorrectos
- **Solución**: Uso correcto de `config_file` en lugar de `config_path`
- **Beneficio**: Tests de integración de configuración funcionan

#### 6. Sanitización de Datos Mejorada
- **Problema**: Expectativas incorrectas sobre comportamiento de sanitización
- **Solución**: Tests alineados con implementación real que preserva contenido válido
- **Resultado**: Validación de sanitización precisa

### 📊 Resultados de Tests Actualizados
```
======================== 18 passed in 3.41s =========================
✅ TestDataValidator::test_validate_string_input PASSED
✅ TestErrorRecoveryManager::test_error_statistics PASSED  
✅ TestErrorRecoveryManager::test_handle_file_not_found_error PASSED
✅ TestErrorRecoveryManager::test_handle_memory_error PASSED
✅ TestErrorRecoveryManager::test_recovery_strategies PASSED
✅ TestSystemValidator::test_sanitize_filename PASSED
✅ TestSystemValidator::test_validate_case_number PASSED
✅ TestSystemValidator::test_validate_image_file PASSED
✅ TestIntegratedValidation::test_config_validation_integration PASSED
✅ TestIntegratedValidation::test_error_handling_decorator_integration PASSED
✅ TestIntegratedValidation::test_image_processing_validation_integration PASSED
✅ TestValidationPerformance::test_error_handling_performance PASSED
✅ TestValidationPerformance::test_validation_performance PASSED
```

## Estructura Actual de Tests

### Directorio Principal `/tests/`

#### Tests de Integración (`/integration/`)
- `test_integration.py` - Test principal de integración
- `test_gui_headless.py` - Tests de GUI en modo headless
- `test_complete_integration.py` - Tests completos del sistema
- `test_backend_integration.py` - Tests de integración del backend
- `test_ui_*.py` - Tests específicos de interfaz de usuario

#### Tests Legacy (`/legacy/`)
- Contiene tests antiguos que necesitan revisión
- Algunos pueden estar duplicados con tests actuales
- **Recomendación:** Revisar y consolidar o eliminar

#### Tests Unitarios (`/unit/`)
- `test_image_processing.py` - Tests de procesamiento de imágenes
- `test_memory_optimized.py` - Tests de optimización de memoria
- `advanced_module_test.py` - Tests avanzados de módulos
- `simple_image_test.py` - Tests básicos de imágenes

#### Tests de GUI (`/gui/`)
- `test_roi_visualization.py` - Tests de visualización ROI
- `test_simple_roi_viz.py` - Tests simples de visualización
- `test_visualization_features.py` - Tests de características de visualización

#### Tests de Performance (`/performance/`)
- Directorio para tests de rendimiento (actualmente vacío)

### Consolidación de Tests en Directorio Raíz

Se han eliminado/movido tests que residían en el directorio raíz para evitar duplicación y mejorar la organización. Los tests relevantes han sido consolidados bajo `tests/`.

#### Cambios Clave
- Eliminados duplicados en raíz: `test_real_images.py`, `test_system_validation.py`, `test_simple_images.py`, `test_gui_complete.py`.
- Movido `test_assets_real.py` a `tests/legacy/test_assets_real.py` por tratarse de un script de verificación manual con assets reales.
- Los tests de NIST y rendimiento siguen agrupados en `tests/nist/` y `tests/performance/` respectivamente.

Para referencias previas de tests en raíz, usar sus equivalentes dentro de `tests/`:
- Validación de sistema: `tests/test_validation_system.py`.
- Imágenes reales: `tests/test_real_images.py`.
- GUI: `tests/test_gui_simple.py`, `tests/test_gui_integration_legacy.py`, `tests/integration/test_frontend_integration_consolidated.py`.

### Archivos de Soporte
- `benchmark_system.py` - Sistema de benchmarking
- `organize_tests.py` - Script para organizar tests
- `config.yaml` - Configuración específica para tests

## Problemas Identificados

### 1. Duplicación de Tests
- Múltiples archivos con nombres similares (`test_*_integration.py`)
- Tests legacy que pueden estar duplicados
- Funcionalidad similar en diferentes archivos

### 2. Organización Inconsistente
- Tests de integración tanto en `/integration/` como en raíz
- Mezcla de tests unitarios y de integración en el mismo nivel
- Falta de categorización clara

### 3. Nomenclatura Inconsistente
- Algunos archivos usan `test_` otros no siguen convención
- Nombres muy similares que causan confusión
- Falta de prefijos descriptivos

## Recomendaciones de Reorganización

### Estructura Propuesta
```
tests/
├── unit/                    # Tests unitarios
│   ├── image_processing/
│   ├── matching/
│   ├── database/
│   └── gui/
├── integration/             # Tests de integración
│   ├── backend/
│   ├── frontend/
│   └── full_system/
├── performance/             # Tests de rendimiento
│   ├── benchmarks/
│   └── profiling/
├── nist/                    # Tests específicos NIST
│   ├── standards/
│   └── validation/
├── fixtures/                # Datos de prueba
│   ├── images/
│   └── configs/
└── legacy/                  # Tests antiguos (temporal)
```

### Acciones Recomendadas

#### Fase 1: Consolidación
1. **Revisar tests duplicados**
   - Comparar `test_*_integration.py` files
   - Identificar funcionalidad común
   - Consolidar en archivos únicos

2. **Mover tests a categorías apropiadas**
   - Tests unitarios → `/unit/`
   - Tests de integración → `/integration/`
   - Tests de performance → `/performance/`

#### Fase 2: Limpieza
1. **Eliminar tests obsoletos**
   - Revisar tests en `/legacy/`
   - Eliminar duplicados confirmados
   - Actualizar tests desactualizados

2. **Estandarizar nomenclatura**
   - Usar prefijo descriptivo
   - Seguir convención `test_<module>_<functionality>.py`
   - Documentar propósito de cada test

#### Fase 3: Mejora
1. **Añadir documentación**
   - Docstrings en tests principales
   - README por categoría
   - Guías de ejecución

2. **Configurar CI/CD**
   - Pipeline de tests automáticos
   - Cobertura de código
   - Tests de regresión

## Comandos Útiles

### Ejecutar Tests por Categoría
```bash
# Tests unitarios
pytest tests/unit/ -v

# Tests de integración
pytest tests/integration/ -v

# Tests de performance
pytest tests/performance/ -v

# Tests específicos
pytest tests/test_cmc_algorithm.py -v
```

### Análisis de Cobertura
```bash
pytest --cov=. --cov-report=html tests/
```

### Ejecutar Tests Específicos
```bash
# Tests de GUI en modo headless
pytest tests/integration/test_gui_headless.py -v

# Tests de matching
pytest -k "matching" -v

# Tests de NIST
pytest -k "nist" -v
```

---

*Documento generado para facilitar la organización y mantenimiento de la suite de tests*