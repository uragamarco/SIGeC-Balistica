# Organización de Tests - SIGeC-Balisticar

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

### Tests en Directorio Raíz

#### Tests de Algoritmos Específicos
- `test_cmc_algorithm.py` - Tests del algoritmo CMC
- `test_bootstrap_confidence.py` - Tests de confianza bootstrap
- `test_matching_analysis.py` - Tests de análisis de matching
- `test_statistical_analysis.py` - Tests de análisis estadístico

#### Tests de Sistemas
- `test_consolidated_system.py` - Tests del sistema consolidado
- `test_unified_config.py` - Tests de configuración unificada
- `test_dependency_manager.py` - Tests del gestor de dependencias
- `test_pipeline_config.py` - Tests de configuración de pipeline

#### Tests de Procesamiento
- `test_image_processing.py` - Tests de procesamiento de imágenes
- `test_memory_optimization.py` - Tests de optimización de memoria
- `test_lbp_cache.py` - Tests de cache LBP

#### Tests de Matching
- `test_improved_matching.py` - Tests de matching mejorado
- `test_optimized_matching.py` - Tests de matching optimizado
- `test_matcher_comparison.py` - Tests de comparación de matchers
- `test_debug_matching.py` - Tests de debug de matching

#### Tests de Integración NIST
- `test_nist_standards.py` - Tests de estándares NIST
- `test_nist_validation.py` - Tests de validación NIST
- `test_nist_real_images.py` - Tests con imágenes reales NIST
- `test_nist_real_images_simple.py` - Tests simples con imágenes NIST
- `test_nist_statistical_integration.py` - Tests de integración estadística NIST

#### Tests de Performance
- `test_performance_benchmarks.py` - Benchmarks de rendimiento
- `test_parallel_performance.py` - Tests de rendimiento paralelo
- `test_parallel_simple.py` - Tests simples de paralelización
- `test_pipeline_performance.py` - Tests de rendimiento de pipeline

#### Tests de Integración Diversos
- `test_basic_integration.py` - Tests básicos de integración
- `test_simple_integration.py` - Tests simples de integración
- `test_complete_integration.py` - Tests completos de integración
- `test_hybrid_integration.py` - Tests de integración híbrida
- `test_integration_chunked.py` - Tests de integración por chunks
- `test_integration_systems.py` - Tests de sistemas de integración

#### Tests Especializados
- `test_real_images.py` - Tests con imágenes reales
- `test_scientific_pipeline.py` - Tests de pipeline científico
- `test_spatial_calibration_integration.py` - Tests de calibración espacial
- `test_compatibility_validation.py` - Tests de validación de compatibilidad
- `test_gui_statistical_integration.py` - Tests de integración GUI-estadística

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