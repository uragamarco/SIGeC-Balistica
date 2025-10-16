# Informe de Estado – `gui/analysis_tab.py`

Fecha: {{AUTO}}

## Descripción del Proceso de Usuario
- Paso 1 – Selección de imágenes: El usuario carga una o más imágenes mediante `ImageSelector`. El stepper marca el paso como completado al tener imágenes válidas.
- Paso 2 – Configuración: El usuario define nivel de análisis (Básico/Intermedio/Avanzado) y parámetros en `ConfigurationLevelsManager` y widgets compartidos (procesamiento, NIST, AFTE, Deep Learning).
- Paso 3 – Ejecución de análisis: Se inicia el análisis con `Start Analysis`. `AnalysisWorker` corre en segundo plano validando imágenes, preprocesando y ejecutando el análisis comparativo.
- Paso 4 – Revisión de resultados: Al completar, se actualiza el panel de resultados y el stepper lleva a la pestaña de resultados. Desde allí se revisan tarjetas, estadísticas y conclusión AFTE.

## Ventajas
- Flujo guiado claro mediante `AnalysisStepper`.
- Separación de responsabilidades: UI en `AnalysisTab`, procesamiento en `AnalysisWorker`.
- Integración con paneles específicos para visualización y resultados.
- Señales para telemetría y performance ya integradas.

## Falencias y Fallos Identificados
- Llamada a `self.image_selector.update_selection(images)` en `on_images_changed`: `ImageSelector` no implementa `update_selection`. Provoca error de atributo.
- Uso de `self.results_panel.display_results(results)` en `on_analysis_completed`: `ResultsPanel` (alias de `DynamicResultsPanel`) no define `display_results`. Causa error al finalizar.
- Cuando solo hay una imagen, el pipeline retorna error simple. Falta manejo guiado para análisis de una imagen (extracción de características, calidad) o exigir 2 imágenes antes del paso 3.
- Ausencia de verificación `hasattr` previa a llamadas de métodos no garantizados.
- El limpiado de resultados depende de paneles con APIs dispares (`visualization_methods.clear_results` vs `ResultsPanel.clear_results`).

## Mejoras Propuestas
- Añadir `display_results` a `DynamicResultsPanel` para aceptar dict/list y generar tarjetas, actualizar estadísticas y conclusión AFTE.
- Reemplazar `update_selection` por: 
  - Añadir `set_images(images: List[str])` a `ImageSelector` y usarlo; o
  - Suprimir la llamada y dejar que la UI se actualice mediante las señales existentes (`imagesChanged`).
- Validar número de imágenes antes de avanzar a análisis: 
  - Si hay una imagen: correr flujo de extracción de características y calidad.
  - Si hay dos o más: correr comparación; en caso contrario, advertir al usuario y bloquear el paso.
- Añadir comprobaciones seguras: `if hasattr(self.results_panel, 'display_results'):` antes de invocar.
- Documentar el contrato de `ResultsPanel` y estandarizarlo para todas las pestañas.

## Estado Actual y Pendientes
- Arquitectura lista para integrar `ScientificPipeline` y módulos del `core`.
- Pendiente alineación de APIs de widgets (`ResultsPanel`, `ImageSelector`).
- Pendiente consolidación de visualizaciones específicas por tipo de resultado.

## Recomendaciones de Implementación
- Implementar `display_results` en `dynamic_results_panel.py` con mapeo:
  - Si `results` es dict con claves `features`, `scores`, `afte_conclusion`, `confidence`, crear tarjetas y actualizar `AFTEConclusionWidget`.
  - Si es lista de dicts con campos `id/similarity/confidence/match_type`, crear tarjetas con color y progreso.
- Agregar `set_images` opcional en `ImageSelector` y usarlo desde `AnalysisTab`.
- Establecer pruebas manuales del flujo con sets de ejemplo y verificar que el stepper cambia correctamente entre pestañas.