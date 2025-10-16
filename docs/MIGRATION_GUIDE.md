---
title: Guía de Migración
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: desarrolladores, devops
toc: true
tags:
  - migración
  - configuración
  - capas
---

# SIGeC-Balisticar - Guía de Migración

## Introducción

Esta guía te ayudará a migrar del sistema de configuración anterior al nuevo **Sistema de Configuración en Capas** implementado en SIGeC-Balisticar v2.0.

## ¿Qué ha cambiado?

### Antes (Sistema Anterior)
- Múltiples archivos de configuración dispersos
- Configuración específica por módulo
- Duplicación de valores
- Difícil mantenimiento

### Ahora (Sistema en Capas)
- **Configuración unificada** en `config/config_layers.yaml`
- **Herencia de configuración** entre entornos
- **Sobrescritura mediante variables de entorno**
- **Gestión centralizada** a través de `UnifiedConfigManager`

## Pasos de Migración

### 1. Backup de Configuraciones Existentes

```bash
# Crear backup de configuraciones actuales
mkdir -p backup_config
cp -r config/ backup_config/
cp *.yaml backup_config/ 2>/dev/null || true
cp *.yml backup_config/ 2>/dev/null || true
```

### 2. Identificar Archivos de Configuración Antiguos

Los siguientes archivos son **obsoletos** y pueden eliminarse después de la migración:

- `config.yaml` (configuración principal antigua)
- `gui_config.yaml` (configuración GUI específica)
- `test_config.yaml` (configuración de pruebas)
- `production_config.yaml` (configuración de producción)
- Cualquier archivo `*_config.yaml` específico por módulo

### 3. Migración Automática

El sistema incluye **migración automática** de configuraciones existentes:

```python
from config.config_manager import get_unified_manager

# El gestor detecta y migra automáticamente configuraciones antiguas
config_manager = get_unified_manager()

# Cargar configuración (migra automáticamente si es necesario)
config = config_manager.load_config("unified", "base")
```

### 4. Verificar Migración

```python
# Verificar que la migración fue exitosa
from config.config_manager import get_unified_manager

config_manager = get_unified_manager()

# Probar acceso a valores migrados
db_host = config_manager.get_config_value("database.host")
gui_theme = config_manager.get_config_value("gui.theme")

print(f"Database Host: {db_host}")
print(f"GUI Theme: {gui_theme}")
```

## Mapeo de Configuraciones

### Configuración de Base de Datos

**Antes:**
```yaml
# config.yaml
database:
  type: "sqlite"
  path: "data/ballistic_db.db"
```

**Ahora:**
```yaml
# config/config_layers.yaml
base:
  database:
    type: "unified"
    host: "localhost"
    port: 5432
    name: "ballistic_db"
```

### Configuración GUI

**Antes:**
```yaml
# gui_config.yaml
theme: "modern"
enable_gpu: true
window_size: [1200, 800]
```

**Ahora:**
```yaml
# config/config_layers.yaml
base:
  gui:
    theme: "modern"
    enable_gpu: true
    window:
      width: 1200
      height: 800
```

### Configuración de Procesamiento de Imágenes

**Antes:**
```yaml
# image_config.yaml
roi_detection: "watershed"
feature_extraction: "orb_sift_hybrid"
```

**Ahora:**
```yaml
# config/config_layers.yaml
base:
  image_processing:
    roi_detection: "watershed"
    feature_extraction: "orb_sift_hybrid"
```

## Actualización de Código

### Importaciones Antiguas vs Nuevas

**Antes:**
```python
# Importaciones antiguas (OBSOLETAS)
from config.config_manager import ConfigManager
from config.gui_config import GUIConfig
from config.production_config import ProductionConfig
```

**Ahora:**
```python
# Nueva importación unificada
from config.config_manager import get_unified_manager
```

### Uso de Configuración

**Antes:**
```python
# Código antiguo (OBSOLETO)
config_manager = ConfigManager()
gui_config = GUIConfig()
prod_config = ProductionConfig()

db_host = config_manager.get("database.host")
theme = gui_config.get("theme")
```

**Ahora:**
```python
# Nuevo código
config_manager = get_unified_manager()

# Cargar configuración por entorno
config = config_manager.load_config("unified", "production")

# Obtener valores específicos
db_host = config_manager.get_config_value("database.host")
theme = config_manager.get_config_value("gui.theme")
```

## Configuración por Entornos

### Entorno de Desarrollo
```python
# Cargar configuración base para desarrollo
config = config_manager.load_config("unified", "base")
```

### Entorno de Testing
```python
# Cargar configuración de testing (hereda de base + overrides)
config = config_manager.load_config("unified", "testing")
```

### Entorno de Producción
```python
# Cargar configuración de producción (hereda de base + overrides)
config = config_manager.load_config("unified", "production")
```

## Variables de Entorno

El nuevo sistema soporta sobrescritura mediante variables de entorno:

```bash
# Configurar entorno
export SIGEC_ENVIRONMENT="production"

# Sobrescribir valores específicos
export SIGEC_DATABASE_HOST="prod-db-server.company.com"
export SIGEC_DATABASE_PORT="5432"
export SIGEC_GUI_THEME="dark"
export SIGEC_LOGGING_LEVEL="INFO"
```

## Validación y Testing

### Validar Configuración
```python
# Validar configuración cargada
is_valid = config_manager.validate_config("unified", "production")
if not is_valid:
    print("Error en la configuración de producción")
```

### Testing de Migración
```python
# Script de testing para verificar migración
def test_migration():
    config_manager = get_unified_manager()
    
    # Probar carga de diferentes entornos
    environments = ["base", "testing", "production"]
    
    for env in environments:
        try:
            config = config_manager.load_config("unified", env)
            print(f"✓ Configuración {env} cargada correctamente")
        except Exception as e:
            print(f"✗ Error en configuración {env}: {e}")
    
    # Probar acceso a valores críticos
    critical_values = [
        "database.host",
        "gui.theme",
        "image_processing.roi_detection",
        "matching.algorithm"
    ]
    
    for value_path in critical_values:
        try:
            value = config_manager.get_config_value(value_path)
            print(f"✓ {value_path}: {value}")
        except Exception as e:
            print(f"✗ Error accediendo {value_path}: {e}")

# Ejecutar test
test_migration()
```

## Limpieza Post-Migración

Una vez verificada la migración exitosa:

```bash
# Eliminar archivos de configuración obsoletos
rm -f config.yaml
rm -f gui_config.yaml
rm -f test_config.yaml
rm -f production_config.yaml

# Mantener backup por seguridad
# NO eliminar backup_config/ hasta estar 100% seguro
```

## Solución de Problemas

### Error: "ConfigType not found"
```python
# Si encuentras este error, asegúrate de usar la nueva API
# INCORRECTO:
# config = config_manager.load_config(ConfigType.UNIFIED)

# CORRECTO:
config = config_manager.load_config("unified", "base")
```

### Error: "Configuration file not found"
```python
# Verificar que config_layers.yaml existe
import os
config_path = "config/config_layers.yaml"
if not os.path.exists(config_path):
    print(f"Archivo de configuración no encontrado: {config_path}")
    # Crear configuración base o restaurar desde backup
```

### Valores de Configuración Faltantes
```python
# Usar valores por defecto para configuraciones faltantes
db_host = config_manager.get_config_value("database.host", "localhost")
gui_theme = config_manager.get_config_value("gui.theme", "default")
```

## Beneficios del Nuevo Sistema

1. **Configuración Unificada**: Un solo archivo para toda la configuración
2. **Herencia**: Los entornos heredan de la configuración base
3. **Flexibilidad**: Sobrescritura fácil mediante variables de entorno
4. **Mantenibilidad**: Gestión centralizada y consistente
5. **Validación**: Validación automática de configuraciones
6. **Migración**: Migración automática desde sistemas anteriores

## Soporte

Si encuentras problemas durante la migración:

1. **Verifica el backup** de tus configuraciones originales
2. **Revisa los logs** para errores específicos
3. **Usa la validación** integrada para identificar problemas
4. **Consulta esta guía** para patrones de migración comunes

Para soporte adicional, consulta la documentación del proyecto o contacta al equipo de desarrollo.