# Plan de Acciones y Tareas Recomendadas (2025)

**Fecha:** Octubre 2025  
**Versión:** 1.0

## Objetivo
Alinear el proyecto SIGeC-Balisticar con una arquitectura unificada, reducir duplicaciones, y fortalecer confiabilidad en producción mediante un plan accionable de corto (Fase 1), medio (Fase 2) y largo plazo (Fase 3).

## Principios
- Una única fuente de verdad para configuración (`config/unified_config.py`).
- Interfaces claras y desacopladas, evitando imports condicionales extensivos.
- Fallbacks centralizados y auditables.
- Documentación viva y probada (tests de configuración y contratos).

## Fase 1: Unificación y Estabilización (0–4 semanas)

### 1. Unificar sistema de configuración
- Acción: Corregir `config/config_manager.py` eliminando referencias a `layered_config_manager` y alinearlo con `UnifiedConfig`.
- Acción: Mantener `config_consolidator.py` como descubridor/migrador, no como gestor en ejecución.
- Acción: Revisar y documentar diferencias entre `unified_config.yaml`, `unified_config_production.yaml`, `unified_config_testing.yaml`.
- Entregables: Guía de configuración unificada, tests de carga de YAML por entorno.

### 2. Consolidar fallbacks
- Acción: Crear `core/fallback_registry.py` (si no existe) centralizado con mapeos explícitos.
- Acción: Migrar uso disperso de fallbacks (`utils/fallback_implementations.py`) hacia registros únicos.
- Entregables: Lista de dependencias críticas con fallback y pruebas unitarias.

### 3. Completar `__init__.py` y contratos
- Acción: Añadir/exportar interfaces públicas estables en módulos núcleo (`core`, `image_processing`, `matching`).
- Entregables: Índices claros y documentación de import estable.

### 4. Limpieza de paths hardcodeados
- Acción: Ejecutar y revisar `scripts/cleanup_hardcoded_paths.py`; eliminar rutas absolutas en módulos críticos.
- Entregables: Informe de rutas corregidas y CI con validación.

## Fase 2: Integración y Validación (4–10 semanas)

### 5. Integrar GUI con configuración unificada
- Acción: Actualizar `gui/settings_dialog.py` y flujos de lectura/escritura para usar `UnifiedConfig`.
- Entregables: Pruebas funcionales GUI para distintos entornos.

### 6. Eliminar duplicaciones de ScientificPipeline
- Acción: Alinear `core/unified_pipeline.py` como implementación oficial; deprecar mocks en `utils/fallback_implementations.py` donde no sean necesarios.
- Entregables: Tests de regresión (`tests/test_scientific_pipeline.py`) pasando con configuración unificada.

### 7. Centralizar configuración de base de datos
- Acción: Confirmar uso único de `DatabaseConfig` desde `config/unified_config.py` en producción y GUI.
- Entregables: Pruebas de conexión y migraciones de configuración.

## Fase 3: Optimización y Observabilidad (10–20 semanas)

### 8. Telemetría y rendimiento
- Acción: Integrar `performance/enhanced_monitoring_system.py` para recomendaciones automáticas.
- Entregables: Panel de métricas y alertas básicas.

### 9. Seguridad y despliegue
- Acción: Validar `production/deployment_validator.py` con políticas de dependencias y fallbacks.
- Entregables: Checklist de despliegue y CI/CD endurecido.

### 10. Documentación y versionado de configuración
- Acción: Publicar `API_DOCUMENTATION.md` actualizado y guías de migración.
- Entregables: Versionado de YAML con cambios trazables.

## Métricas de Éxito
- Reducción de archivos de configuración activos a 1 fuente de verdad.
- 0 imports rotos en CI; 0 referencias a gestores inexistentes.
- Tests de pipelines y GUI pasando en todos los entornos.
- Disminución de warnings de `deployment_validator` y del uso de fallbacks.

## Riesgos y Mitigaciones
- Riesgo: Roturas por cambio de configuración. Mitigación: Tests y migración guiada.
- Riesgo: Dependencias no disponibles. Mitigación: Fallbacks centralizados y mensajes claros.
- Riesgo: Divergencia GUI. Mitigación: Contratos de lectura/escritura validados y revisiones.

## Próximos Pasos inmediatos
- Confirmar referencias rotas en `config_manager.py` y parchearlas.
- Alinear producción y testing con `UnifiedConfig` efectivo.
- Generar inventario de archivos obsoletos para borrado controlado.