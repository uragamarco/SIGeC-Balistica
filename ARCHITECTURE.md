# Arquitectura del Sistema SIGeC-Balisticar

## Visión General

SIGeC-Balisticar implementa una arquitectura modular enfocada en análisis balístico forense, con separación clara de responsabilidades, configuración unificada y una GUI integrada en PyQt5.

## Mapa de Componentes

### 1. Core Pipeline (`core/`)
- `unified_pipeline.py`: Orquestador principal del procesamiento y flujo científico.
- `pipeline_config.py`: Configuraciones y niveles del pipeline.
- `error_handler.py`: Manejo de errores unificado.
- `intelligent_cache.py`: Cache inteligente para resultados y datos intermedios.
- `telemetry_system.py`: Telemetría y trazabilidad.
- `performance_monitor.py`: Métricas de rendimiento.
- `notification_system.py`: Notificaciones internas.
- `fallback_system.py`: Comportamientos de respaldo.

### 2. Procesamiento de Imágenes (`image_processing/`)
- `lazy_loading.py`: Carga diferida/optimizada de imágenes.
- `lbp_cache.py`: Cache y utilidades LBP.

### 3. Algoritmos de Matching (`matching/`)
- `unified_matcher.py`: Matching de características unificado (SIFT/ORB y otros), integrado con el pipeline.
- `cmc_algorithm.py`: Implementación del algoritmo CMC (Congruent Matching Cells) conforme NIST.
- `bootstrap_similarity.py`: Análisis de similitud con bootstrap para intervalos de confianza.

### 4. Base de Datos (`database/`)
- `unified_database.py`: Acceso y operaciones de base de datos unificada.
- `vector_db.py`: Soporte para base de datos vectorial.

### 5. GUI (`gui/`)
- `main_window.py`: Ventana principal y coordinación de tabs.
- Tabs y componentes: `analysis_tab.py`, `comparison_tab.py`, `database_tab.py`, `reports_tab.py`, `settings_dialog.py` y widgets asociados.

### 6. Configuración (`config/`)
- `unified_config.py`: API de configuración unificada con archivos YAML.
- `unified_config.yaml`: Configuración base (más variantes por entorno).
- `config_layers.yaml`: Material legacy de referencia (no usado en runtime actual).

### 7. Estándares NIST (`nist_standards/`, `common/`)
- `nist_standards/nist_schema.py`: Esquema y metadatos NIST.
- `common/nist_integration.py`: Integración y utilidades NIST.
- `common/statistical_core.py`: Núcleo estadístico.

### 8. Deep Learning (`deep_learning/`)
- Estructura preparada (`config/`, `models/`, `utils/`, `testing/`), pendiente de integración directa con el pipeline.

## Patrones de Diseño

### Factory
- `patterns/matcher_factory.py`: Registro/creación de algoritmos de matching.
- `patterns/preprocessor_factory.py`: Registro/creación de preprocesadores.

### Strategy
- Algoritmos y preprocesadores configurables mediante estrategias.

### Interface Segregation
- `interfaces/matcher_interfaces.py`: Interfaz `IFeatureMatcher` para matchers.
  (Otras interfaces se pueden añadir conforme avance la modularización.)

## Flujo de Procesamiento

1. Carga de imágenes (`image_processing/lazy_loading.py`)
2. Preprocesamiento/Cache LBP (`image_processing/lbp_cache.py`)
3. Extracción y matching de características (`matching/unified_matcher.py`)
4. Cálculo CMC y análisis de similitud (`cmc_algorithm.py`, `bootstrap_similarity.py`)
5. Análisis estadístico y validación NIST (`common/statistical_core.py`, `common/nist_integration.py`)
6. Persistencia/consulta en BD (`database/unified_database.py`, `database/vector_db.py`)
7. Telemetría, rendimiento y notificaciones (`core/*`)

## Extensibilidad

### Agregar nuevo algoritmo de matching
- Implementar `IFeatureMatcher` (ver `interfaces/matcher_interfaces.py`).
- Registrar en `patterns/matcher_factory.py`.
- Añadir parámetros en YAML (`config/unified_config.yaml`).
- Crear tests unitarios e integración bajo `tests/`.

### Agregar preprocesador de imágenes
- Implementar la clase correspondiente y registrar en `patterns/preprocessor_factory.py`.
- Integrar con pipeline y cache según necesidad.

## Testing y Validación

- Configuración de pytest: `pytest.ini` con marcadores `unit`, `integration`, `gui`, `performance`, `nist`.
- Suite consolidada en `tests/` (unitarios, integración, GUI y legacy controlado).
- Consejos: exportar `PYTHONPATH=.` para evitar `ImportError`, y usar `QT_QPA_PLATFORM=offscreen` en entornos headless.

## Estado Actual y Pendientes

- Estructura de tests consolidada; duplicados de raíz eliminados.
- Configuración unificada activa; material legacy documentado y en proceso de retiro controlado.
- Deep Learning: estructura presente, integración con el pipeline pendiente.
- Exports `__init__.py`: revisar y estandarizar en módulos con múltiples clases públicas.
- Verificar y ajustar `tests/test_dependency_manager.py` (un fallo detectado en ejecución focalizada).

## Métricas

- Las métricas de cobertura y rendimiento se recalcularán tras la consolidación de tests y limpieza de legacy.