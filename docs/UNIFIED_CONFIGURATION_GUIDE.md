---
title: Guía de Configuración Unificada
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: desarrolladores
toc: true
tags:
  - configuración
  - entornos
  - validación
---

# SIGeC-Balisticar - Guía de Configuración Unificada

## Objetivo

Explica cómo usar `UnifiedConfig` y `UnifiedConfigManager` para cargar, validar, modificar y guardar configuraciones por entorno (desarrollo, testing, producción).

## Archivos por entorno

- Desarrollo: `config/unified_config.yaml`
- Testing: `config/unified_config_testing.yaml`
- Producción: `config/unified_config_production.yaml`

La selección de entorno puede hacerse explícitamente con el enum `Environment` o mediante la variable `SIGeC-Balistica_ENV` (`dev`, `test`, `prod`).

## Uso básico

```python
from config.unified_config import get_unified_config, Environment

# Carga explícita por entorno
cfg = get_unified_config(environment=Environment.DEVELOPMENT)

# Acceso tipado
db_path = cfg.database.sqlite_path
matcher = cfg.matching.matcher_type

# Validación
cfg.validate()  # Lanza ConfigValidationError con detalles si hay errores

# Guardar
cfg.save_config()
```

## API de compatibilidad (manager)

```python
from config.config_manager import get_unified_manager

cm = get_unified_manager()

# Cargar y leer
cm.load_config("unified", "production")
host = cm.get_config_value("database.sqlite_path")

# Modificar y persistir
cm.set_config_value("logging.level", "INFO")
cm.save_config("unified", "production")

# Validación
errors = cm.validate_config("unified", "production")
if errors:
    for e in errors:
        print(f"- {e}")
```

## Enums y YAML

`UnifiedConfig.save_config()` serializa `matching.algorithm` y `matching.level` como strings para evitar anotaciones específicas de Python.

Valores permitidos:
- `matching.algorithm`: `ORB`, `SIFT`, `AKAZE`, `BRISK`, `KAZE`, `CMC`
- `matching.level`: `basic`, `standard`, `advanced`

Al cargar, el sistema convierte automáticamente estos strings a enums internos.

## Buenas prácticas

- Mantén los YAML de producción sin anotaciones `!!python/*`.
- Usa `cfg.validate()` antes de guardar cambios críticos.
- Centraliza lecturas/escrituras con `UnifiedConfigManager` en módulos que aún dependan de APIs legacy.
- Usa rutas relativas a `project_root` y convierte a absolutas con `cfg.get_absolute_path()` cuando sea necesario.

## Troubleshooting

- Errores de validación: revisa el listado por sección en la excepción `ConfigValidationError` o el array retornado por `config_manager.validate_config()`.
- Entorno incorrecto: verifica `SIGeC-Balistica_ENV` y alias (`dev`, `test`, `prod`).
- Serialización de enums: si ves `!!python/object/apply` en producción, edita el YAML para dejar strings y vuelve a guardar con `cfg.save_config()`.