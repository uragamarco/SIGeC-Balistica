---
title: Guía del Sistema de Validación
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: QA, desarrolladores
toc: true
tags:
  - validación
  - manejo_de_errores
  - testing
  - nist
---

# SIGeC-Balisticar - Guía del Sistema de Validación

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Componentes Principales](#componentes-principales)
4. [Uso del Sistema](#uso-del-sistema)
5. [Configuración](#configuración)
6. [Manejo de Errores](#manejo-de-errores)
7. [Testing](#testing)
8. [Mejores Prácticas](#mejores-prácticas)

---

## Visión General

El Sistema de Validación de SIGeC-Balistica proporciona un framework robusto y extensible para la validación de datos, manejo de errores y recuperación automática. Está diseñado para garantizar la integridad de los datos y la confiabilidad del sistema en entornos forenses críticos.

### Características Principales

- **Validación Tipada**: Soporte para múltiples tipos de datos con reglas específicas
- **Manejo de Errores Inteligente**: Estrategias de recuperación automática basadas en contexto
- **Integración Transparente**: Decoradores y patrones que se integran sin esfuerzo
- **Rendimiento Optimizado**: Validación eficiente con cache y paralelización
- **Estándares NIST**: Cumplimiento con protocolos forenses establecidos

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE VALIDACIÓN                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ DataValidator│  │ErrorHandler │  │SystemValidator│ │ Utils   │ │
│  │             │  │             │  │             │  │         │ │
│  │ • Schemas   │  │ • Recovery  │  │ • Files     │  │ • Cache │ │
│  │ • Rules     │  │ • Strategies│  │ • Images    │  │ • Logs  │ │
│  │ • Types     │  │ • Context   │  │ • Cases     │  │ • Stats │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. DataValidator (`core/data_validator.py`)

El validador principal que maneja esquemas, reglas y tipos de datos.

#### Clases Principales:
- `DataValidator`: Clase principal de validación
- `ValidationRule`: Definición de reglas de validación
- `ValidationResult`: Resultado de operaciones de validación
- `DataType`: Enumeración de tipos de datos soportados

#### Ejemplo de Uso:
```python
from core.data_validator import get_data_validator, ValidationRule, DataType

# Obtener instancia del validador
validator = get_data_validator()

# Definir esquema de validación
schema = {
    "case_number": ValidationRule(
        data_type=DataType.STRING,
        required=True,
        min_length=3,
        max_length=20,
        pattern=r'^[A-Z0-9\-_]+$'
    ),
    "investigator": ValidationRule(
        data_type=DataType.STRING,
        required=True,
        min_length=2,
        max_length=50
    )
}

# Registrar esquema
validator.register_schema("case_data", schema)

# Validar datos
data = {
    "case_number": "CASE-2024-001",
    "investigator": "Dr. Smith"
}

result = validator.validate_data("case_data", data)
if result.is_valid:
    print("Datos válidos")
else:
    print(f"Errores: {result.errors}")
```

### 2. ErrorHandler (`core/error_handler.py`)

Sistema inteligente de manejo de errores con recuperación automática.

#### Clases Principales:
- `ErrorRecoveryManager`: Gestor principal de errores
- `RecoveryStrategy`: Estrategias de recuperación disponibles
- `ErrorContext`: Contexto de error para decisiones inteligentes

#### Estrategias de Recuperación:
- `RETRY`: Reintentar operación
- `FALLBACK`: Usar método alternativo
- `GRACEFUL_DEGRADATION`: Funcionalidad reducida
- `USER_INTERVENTION`: Requiere intervención manual

#### Ejemplo de Uso:
```python
from core.error_handler import get_error_manager, with_error_handling

# Decorador para manejo automático
@with_error_handling(
    component="image_processing",
    operation="load_image",
    max_retries=3
)
def load_image(image_path):
    # Lógica de carga de imagen
    pass

# Uso directo del gestor
error_manager = get_error_manager()
try:
    # Operación que puede fallar
    result = risky_operation()
except Exception as e:
    recovery_strategy = error_manager.handle_error(e, context={
        "component": "matching",
        "operation": "feature_extraction"
    })
```

### 3. SystemValidator (`utils/validators.py`)

Validador especializado para elementos del sistema forense.

#### Funcionalidades:
- Validación de números de caso
- Validación de archivos de imagen
- Sanitización de nombres de archivo
- Validación de datos de investigador

#### Ejemplo de Uso:
```python
from utils.validators import SystemValidator

validator = SystemValidator()

# Validar número de caso
is_valid = validator.validate_case_number("CASE-2024-001")

# Validar archivo de imagen
result = validator.validate_image_file("/path/to/evidence.jpg")

# Sanitizar nombre de archivo
safe_name = validator.sanitize_filename("evidence <script>.jpg")
```

---

## Uso del Sistema

### Validación Básica de Datos

```python
# 1. Obtener validador
validator = get_data_validator()

# 2. Definir reglas
rules = {
    "field_name": ValidationRule(
        data_type=DataType.STRING,
        required=True,
        min_length=1,
        max_length=100
    )
}

# 3. Registrar esquema
validator.register_schema("my_schema", rules)

# 4. Validar datos
result = validator.validate_data("my_schema", {"field_name": "value"})
```

### Manejo de Errores Avanzado

```python
# Configurar contexto de error
context = ErrorContext(
    component="image_processing",
    operation="feature_extraction",
    severity=ErrorSeverity.HIGH,
    user_data={"image_path": "/path/to/image.jpg"}
)

# Manejar error con contexto
try:
    process_image()
except Exception as e:
    strategy = error_manager.handle_error(e, context)
    if strategy == RecoveryStrategy.RETRY:
        # Lógica de reintento
        pass
```

---

## Configuración

### Configuración del Validador

```python
# En config/unified_config.py
validation_config = {
    "max_string_length": 1000,
    "enable_sanitization": True,
    "strict_mode": False,
    "cache_results": True
}
```

### Configuración del Manejo de Errores

```python
# En config/unified_config.py
error_handling_config = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "enable_notifications": True,
    "log_level": "INFO"
}
```

---

## Testing

### Estructura de Tests

El sistema incluye tests comprehensivos organizados en:

- **Tests Unitarios**: Validación de componentes individuales
- **Tests de Integración**: Validación de interacciones entre componentes
- **Tests de Rendimiento**: Validación de performance bajo carga
- **Tests de Recuperación**: Validación de estrategias de error

### Ejecutar Tests

```bash
# Tests completos del sistema de validación
python3 -m pytest tests/test_validation_system.py -v

# Tests específicos
python3 -m pytest tests/test_validation_system.py::TestDataValidator -v

# Tests con cobertura
python3 -m pytest tests/test_validation_system.py --cov=core --cov-report=html
```

### Resultados Esperados

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

---

## Mejores Prácticas

### 1. Definición de Esquemas

```python
# ✅ Buena práctica: Esquemas específicos y reutilizables
FORENSIC_CASE_SCHEMA = {
    "case_number": ValidationRule(
        data_type=DataType.STRING,
        required=True,
        pattern=r'^[A-Z0-9\-_]{3,20}$',
        description="Número único de caso forense"
    ),
    "evidence_type": ValidationRule(
        data_type=DataType.STRING,
        required=True,
        allowed_values=["bullet", "cartridge", "firearm"],
        description="Tipo de evidencia balística"
    )
}

# ❌ Mala práctica: Validación ad-hoc sin esquema
if len(case_number) < 3 or not case_number.isalnum():
    raise ValueError("Invalid case number")
```

### 2. Manejo de Errores

```python
# ✅ Buena práctica: Uso de decoradores y contexto
@with_error_handling(
    component="image_processing",
    operation="feature_extraction",
    max_retries=3
)
def extract_features(image_path):
    return process_image(image_path)

# ❌ Mala práctica: Try-catch genérico
try:
    result = process_image(image_path)
except Exception:
    return None
```

### 3. Validación de Performance

```python
# ✅ Buena práctica: Validación con límites de tiempo
def test_validation_performance():
    start_time = time.time()
    
    for i in range(1000):
        validator.validate_data("schema", test_data)
    
    elapsed = time.time() - start_time
    assert elapsed < 5.0, f"Validation too slow: {elapsed}s"
```

### 4. Logging y Monitoreo

```python
# ✅ Buena práctica: Logging estructurado
logger.info("Validation completed", extra={
    "schema": schema_name,
    "records_processed": len(data),
    "validation_time": elapsed_time,
    "errors_found": len(result.errors)
})
```

---

## Actualizaciones Recientes

### Diciembre 2024 - Correcciones Críticas

1. **Importaciones Corregidas**: Agregadas importaciones faltantes de `ValidationRule` y `DataType`
2. **Métodos Actualizados**: Migración de `validate_input` a `validate_data` con esquemas
3. **Estadísticas Alineadas**: Corrección de claves de estadísticas de error
4. **Estrategias Corregidas**: Alineación de estrategias de recuperación con implementación
5. **Configuración Unificada**: Corrección de parámetros de constructor
6. **Sanitización Mejorada**: Tests alineados con comportamiento real

### Próximas Mejoras

- Validación asíncrona para operaciones de larga duración
- Integración con sistema de métricas en tiempo real
- Soporte para validación de esquemas JSON Schema
- Validación de datos de imagen con OpenCV
- Integración con sistema de notificaciones

---

## Soporte y Contribución

Para reportar problemas o contribuir al sistema de validación:

1. **Issues**: Usar el sistema de issues del repositorio
2. **Tests**: Agregar tests para nuevas funcionalidades
3. **Documentación**: Mantener documentación actualizada
4. **Estándares**: Seguir las mejores prácticas establecidas

---

*Documento actualizado: Diciembre 2024*
*Versión del Sistema: 2.0*