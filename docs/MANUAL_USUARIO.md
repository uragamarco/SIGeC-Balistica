---
title: Manual de Usuario
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: usuarios finales, analistas forenses
toc: true
tags:
  - usuario
  - gui
  - cli
  - análisis
---

# SIGeC-Balisticar - Manual de Usuario

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Primeros Pasos](#primeros-pasos)
3. [Interfaz de Usuario](#interfaz-de-usuario)
4. [Análisis de Imágenes](#análisis-de-imágenes)
5. [Gestión de Base de Datos](#gestión-de-base-de-datos)
6. [Comparación de Muestras](#comparación-de-muestras)
7. [Configuración del Sistema](#configuración-del-sistema)
8. [Reportes y Resultados](#reportes-y-resultados)
9. [Solución de Problemas](#solución-de-problemas)
10. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## Introducción

### ¿Qué es SIGeC-Balistica?

SIGeC-Balistica es un sistema integral de gestión y comparación balística que permite:

- **Análisis automatizado** de imágenes balísticas
- **Comparación inteligente** de muestras usando algoritmos avanzados
- **Gestión centralizada** de base de datos de evidencias
- **Cumplimiento** con estándares NIST
- **Interfaz intuitiva** para usuarios técnicos y no técnicos

### Características Principales

✅ **Procesamiento de Imágenes**: Algoritmos LBP, SIFT, ORB para análisis de características  
✅ **Base de Datos Integrada**: Almacenamiento seguro y búsqueda eficiente  
✅ **Comparación Automática**: Cálculo de similitud con múltiples métricas  
✅ **Estándares NIST**: Cumplimiento con normativas internacionales  
✅ **Interfaz Gráfica**: GUI intuitiva desarrollada en PyQt5  
✅ **Reportes Detallados**: Generación automática de informes técnicos  

### Requisitos del Usuario

- **Conocimientos básicos** de informática
- **Comprensión** de conceptos balísticos básicos
- **Acceso** a imágenes digitales de muestras balísticas
- **Permisos** de usuario en el sistema

---

## Primeros Pasos

### Acceso al Sistema

#### Inicio de Sesión Web
1. Abra su navegador web
2. Navegue a la dirección del sistema: `https://su-servidor.com`
3. Ingrese sus credenciales:
   - **Usuario**: Su nombre de usuario asignado
   - **Contraseña**: Su contraseña segura
4. Haga clic en "Iniciar Sesión"

#### Aplicación de Escritorio
1. Ejecute la aplicación desde el menú de inicio
2. O desde la línea de comandos: `python main.py`
3. La interfaz gráfica se abrirá automáticamente

### Primer Uso

Al iniciar por primera vez:

1. **Verificar Configuración**: El sistema verificará automáticamente la configuración
2. **Cargar Módulos**: Se inicializarán todos los módulos necesarios
3. **Conectar Base de Datos**: Se establecerá conexión con la base de datos
4. **Mostrar Panel Principal**: Se presentará la interfaz principal

---

## Interfaz de Usuario

### Panel Principal

La interfaz principal está dividida en varias secciones:

```
┌─────────────────────────────────────────────────────────────┐
│ [Archivo] [Editar] [Ver] [Herramientas] [Ayuda]            │
├─────────────────────────────────────────────────────────────┤
│ 📁 Análisis │ 🗄️ Base de Datos │ ⚖️ Comparación │ ⚙️ Config │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Área de Trabajo Principal]                               │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Estado: Listo │ Progreso: ████████░░ 80% │ Tiempo: 00:02:15 │
└─────────────────────────────────────────────────────────────┘
```

### Pestañas Principales

#### 1. 📁 Pestaña de Análisis
- **Carga de imágenes**: Arrastrar y soltar o seleccionar archivos
- **Configuración de algoritmos**: Selección de métodos de análisis
- **Vista previa**: Visualización de imágenes cargadas
- **Resultados**: Mostrar características extraídas

#### 2. 🗄️ Pestaña de Base de Datos
- **Explorador de registros**: Lista de todas las muestras
- **Búsqueda avanzada**: Filtros por fecha, tipo, características
- **Gestión de registros**: Agregar, editar, eliminar muestras
- **Importar/Exportar**: Funciones de respaldo y migración

#### 3. ⚖️ Pestaña de Comparación
- **Selección de muestras**: Elegir elementos a comparar
- **Configuración de métricas**: Ajustar parámetros de similitud
- **Matriz de resultados**: Visualización de comparaciones
- **Análisis estadístico**: Gráficos y métricas detalladas

#### 4. ⚙️ Pestaña de Configuración
- **Configuración general**: Parámetros del sistema
- **Algoritmos**: Ajustes de procesamiento
- **Base de datos**: Configuración de conexión
- **Interfaz**: Personalización de la GUI

---

## Análisis de Imágenes

### Carga de Imágenes

#### Métodos de Carga

**1. Arrastrar y Soltar**
- Arrastre archivos de imagen directamente al área de trabajo
- Formatos soportados: JPG, PNG, TIFF, BMP
- Múltiples archivos simultáneamente

**2. Selector de Archivos**
- Haga clic en "Cargar Imagen" o "Abrir Archivo"
- Navegue hasta la ubicación de sus archivos
- Seleccione una o múltiples imágenes

**3. Desde Cámara** (si está disponible)
- Conecte una cámara digital
- Seleccione "Capturar desde Cámara"
- Configure parámetros de captura

#### Formatos Soportados

| Formato | Extensión | Descripción |
|---------|-----------|-------------|
| JPEG | .jpg, .jpeg | Formato comprimido estándar |
| PNG | .png | Formato sin pérdida |
| TIFF | .tif, .tiff | Formato profesional |
| BMP | .bmp | Formato sin compresión |

### Configuración de Análisis

#### Selección de Algoritmos

**1. LBP (Local Binary Patterns)**
- **Uso**: Análisis de texturas y patrones locales
- **Ventajas**: Rápido, robusto a cambios de iluminación
- **Parámetros**:
  - Radio: 1-3 (recomendado: 2)
  - Puntos: 8-24 (recomendado: 16)

**2. SIFT (Scale-Invariant Feature Transform)**
- **Uso**: Detección de características distintivas
- **Ventajas**: Invariante a escala y rotación
- **Parámetros**:
  - Número de características: 100-1000
  - Umbral de contraste: 0.03-0.1

**3. ORB (Oriented FAST and Rotated BRIEF)**
- **Uso**: Detección rápida de puntos clave
- **Ventajas**: Computacionalmente eficiente
- **Parámetros**:
  - Número de características: 500-2000
  - Factor de escala: 1.2-2.0

#### Configuración Avanzada

```
┌─── Configuración de Análisis ───┐
│                                 │
│ Algoritmo: [LBP ▼]             │
│                                 │
│ Preprocesamiento:               │
│ ☑ Normalizar iluminación        │
│ ☑ Reducir ruido                 │
│ ☑ Mejorar contraste             │
│                                 │
│ Parámetros LBP:                 │
│ Radio: [2    ] Puntos: [16   ]  │
│                                 │
│ Región de Interés:              │
│ ☐ Seleccionar manualmente       │
│ ☑ Detección automática          │
│                                 │
│ [Aplicar] [Restablecer]         │
└─────────────────────────────────┘
```

### Proceso de Análisis

#### Pasos del Análisis

1. **Carga de Imagen**
   ```
   📁 Cargando imagen... ████████████████████████████████ 100%
   ✅ Imagen cargada: muestra_001.jpg (1920x1080, 2.3 MB)
   ```

2. **Preprocesamiento**
   ```
   🔧 Preprocesando imagen...
   ├─ Normalizando iluminación... ✅
   ├─ Reduciendo ruido... ✅
   └─ Mejorando contraste... ✅
   ```

3. **Extracción de Características**
   ```
   🔍 Extrayendo características LBP...
   ├─ Calculando patrones locales... ████████████████ 100%
   ├─ Generando histograma... ✅
   └─ Características extraídas: 256 dimensiones
   ```

4. **Almacenamiento**
   ```
   💾 Guardando en base de datos...
   ├─ ID asignado: BAL_2024_001
   ├─ Metadatos guardados... ✅
   └─ Características almacenadas... ✅
   ```

### Visualización de Resultados

#### Panel de Resultados

```
┌─── Resultados del Análisis ─────────────────────────────────┐
│                                                             │
│ Imagen Original          │ Imagen Procesada                │
│ ┌─────────────────────┐  │ ┌─────────────────────────────┐ │
│ │                     │  │ │                             │ │
│ │   [Imagen Original] │  │ │   [Imagen con Características] │
│ │                     │  │ │                             │ │
│ └─────────────────────┘  │ └─────────────────────────────┘ │
│                          │                                 │
│ Información:             │ Características:                │
│ • Archivo: muestra_001   │ • Algoritmo: LBP                │
│ • Tamaño: 1920x1080     │ • Dimensiones: 256              │
│ • Formato: JPEG         │ • Tiempo: 2.3s                 │
│ • Fecha: 2024-01-15     │ • Calidad: Alta                 │
│                          │                                 │
│ [Guardar] [Exportar] [Comparar] [Eliminar]                │
└─────────────────────────────────────────────────────────────┘
```

---

## Gestión de Base de Datos

### Explorador de Registros

#### Vista Principal

La pestaña de Base de Datos muestra todos los registros almacenados:

```
┌─── Base de Datos - SIGeC-Balistica ─────────────────────────┐
│                                                             │
│ [🔍 Buscar...] [📅 Filtrar por fecha] [🏷️ Filtrar por tipo] │
│                                                             │
│ ID          │ Nombre        │ Fecha      │ Tipo    │ Estado │
│─────────────┼───────────────┼────────────┼─────────┼────────│
│ BAL_2024_001│ Muestra_001   │ 2024-01-15 │ Casquillo│ ✅     │
│ BAL_2024_002│ Muestra_002   │ 2024-01-15 │ Proyectil│ ✅     │
│ BAL_2024_003│ Muestra_003   │ 2024-01-16 │ Casquillo│ ⏳     │
│ BAL_2024_004│ Muestra_004   │ 2024-01-16 │ Proyectil│ ❌     │
│                                                             │
│ Total: 4 registros │ Seleccionados: 1 │ Página: 1 de 1     │
│                                                             │
│ [➕ Nuevo] [✏️ Editar] [🗑️ Eliminar] [📤 Exportar]          │
└─────────────────────────────────────────────────────────────┘
```

#### Estados de Registro

- ✅ **Procesado**: Análisis completado exitosamente
- ⏳ **Procesando**: Análisis en progreso
- ❌ **Error**: Error durante el procesamiento
- 📋 **Pendiente**: Esperando procesamiento

### Búsqueda y Filtros

#### Búsqueda Básica
- **Texto libre**: Buscar en nombre, descripción, notas
- **ID específico**: Buscar por identificador único
- **Coincidencia parcial**: Búsqueda con comodines (*)

#### Filtros Avanzados

```
┌─── Filtros Avanzados ───┐
│                         │
│ Fecha:                  │
│ Desde: [2024-01-01]     │
│ Hasta: [2024-01-31]     │
│                         │
│ Tipo de Muestra:        │
│ ☑ Casquillos            │
│ ☑ Proyectiles           │
│ ☐ Armas                 │
│                         │
│ Estado:                 │
│ ☑ Procesados            │
│ ☐ Pendientes            │
│ ☐ Con errores           │
│                         │
│ Algoritmo:              │
│ ☑ LBP                   │
│ ☑ SIFT                  │
│ ☑ ORB                   │
│                         │
│ [Aplicar] [Limpiar]     │
└─────────────────────────┘
```

### Gestión de Registros

#### Agregar Nuevo Registro

1. **Haga clic en "➕ Nuevo"**
2. **Complete el formulario**:

```
┌─── Nuevo Registro ──────────────────────────────────────────┐
│                                                             │
│ Información Básica:                                         │
│ Nombre: [Muestra_005________________]                       │
│ Tipo: [Casquillo ▼]                                        │
│ Descripción: [________________________________]            │
│                                                             │
│ Imagen:                                                     │
│ [📁 Seleccionar Archivo...] muestra_005.jpg                │
│                                                             │
│ Metadatos:                                                  │
│ Caso: [CASO_2024_001_______________]                        │
│ Investigador: [Juan Pérez__________]                        │
│ Fecha Recolección: [2024-01-16]                            │
│                                                             │
│ Configuración de Análisis:                                  │
│ Algoritmo: [LBP ▼]                                         │
│ ☑ Procesar automáticamente                                  │
│                                                             │
│ [💾 Guardar] [❌ Cancelar]                                   │
└─────────────────────────────────────────────────────────────┘
```

3. **Haga clic en "💾 Guardar"**
4. **El sistema procesará automáticamente la imagen**

#### Editar Registro Existente

1. **Seleccione un registro** en la lista
2. **Haga clic en "✏️ Editar"**
3. **Modifique los campos necesarios**
4. **Guarde los cambios**

⚠️ **Nota**: No se pueden modificar las características extraídas, solo los metadatos.

#### Eliminar Registros

1. **Seleccione uno o más registros**
2. **Haga clic en "🗑️ Eliminar"**
3. **Confirme la acción**:

```
┌─── Confirmar Eliminación ───┐
│                             │
│ ⚠️ ¿Está seguro de eliminar │
│    los siguientes registros?│
│                             │
│ • BAL_2024_003              │
│ • BAL_2024_004              │
│                             │
│ Esta acción no se puede     │
│ deshacer.                   │
│                             │
│ [🗑️ Eliminar] [❌ Cancelar]  │
└─────────────────────────────┘
```

### Importar y Exportar

#### Importar Datos

**Formatos Soportados**:
- **CSV**: Metadatos con rutas de imágenes
- **JSON**: Formato completo del sistema
- **XML**: Formato estándar de intercambio

**Proceso de Importación**:
1. Haga clic en "📥 Importar"
2. Seleccione el archivo
3. Configure las opciones de mapeo
4. Revise la vista previa
5. Confirme la importación

#### Exportar Datos

**Opciones de Exportación**:
- **Metadatos únicamente**: CSV con información básica
- **Datos completos**: JSON con características
- **Reporte PDF**: Documento formateado
- **Imágenes**: Archivo ZIP con imágenes originales

---

## Comparación de Muestras

### Selección de Muestras

#### Comparación Individual

1. **Vaya a la pestaña "⚖️ Comparación"**
2. **Seleccione "Comparación Individual"**
3. **Elija las muestras a comparar**:

```
┌─── Comparación Individual ──────────────────────────────────┐
│                                                             │
│ Muestra A:                    │ Muestra B:                  │
│ ┌─────────────────────────┐   │ ┌─────────────────────────┐ │
│ │ [Seleccionar Muestra A] │   │ │ [Seleccionar Muestra B] │ │
│ └─────────────────────────┘   │ └─────────────────────────┘ │
│                               │                             │
│ ID: BAL_2024_001              │ ID: BAL_2024_002            │
│ Tipo: Casquillo               │ Tipo: Proyectil            │
│ Fecha: 2024-01-15             │ Fecha: 2024-01-15          │
│                               │                             │
│ Configuración:                                              │
│ Métrica: [Similitud Coseno ▼]                              │
│ Umbral: [0.80_____] (80%)                                  │
│                                                             │
│ [🔍 Comparar] [🔄 Intercambiar] [❌ Limpiar]                │
└─────────────────────────────────────────────────────────────┘
```

#### Comparación Masiva

Para comparar múltiples muestras simultáneamente:

1. **Seleccione "Comparación Masiva"**
2. **Elija el conjunto de muestras**
3. **Configure los parámetros**
4. **Inicie el proceso**

### Métricas de Similitud

#### Tipos de Métricas

**1. Similitud Coseno**
- **Rango**: 0.0 - 1.0
- **Interpretación**: 1.0 = idénticas, 0.0 = completamente diferentes
- **Uso**: Recomendado para características LBP

**2. Distancia Euclidiana**
- **Rango**: 0.0 - ∞
- **Interpretación**: 0.0 = idénticas, mayor valor = más diferentes
- **Uso**: Efectivo para características SIFT/ORB

**3. Correlación de Pearson**
- **Rango**: -1.0 - 1.0
- **Interpretación**: 1.0 = correlación perfecta, -1.0 = anti-correlación
- **Uso**: Análisis estadístico avanzado

#### Configuración de Umbrales

```
┌─── Configuración de Umbrales ───┐
│                                 │
│ Similitud Mínima:               │
│ Alta: [0.90] (90%)              │
│ Media: [0.70] (70%)             │
│ Baja: [0.50] (50%)              │
│                                 │
│ Clasificación Automática:       │
│ ☑ Marcar coincidencias altas    │
│ ☑ Alertar coincidencias medias  │
│ ☐ Ignorar coincidencias bajas   │
│                                 │
│ [Aplicar] [Restablecer]         │
└─────────────────────────────────┘
```

### Resultados de Comparación

#### Matriz de Similitud

```
┌─── Matriz de Similitud ─────────────────────────────────────┐
│                                                             │
│        │ BAL_001 │ BAL_002 │ BAL_003 │ BAL_004 │ BAL_005   │
│────────┼─────────┼─────────┼─────────┼─────────┼───────────│
│ BAL_001│  1.00   │  0.23   │  0.87   │  0.45   │  0.12     │
│ BAL_002│  0.23   │  1.00   │  0.34   │  0.78   │  0.56     │
│ BAL_003│  0.87   │  0.34   │  1.00   │  0.41   │  0.19     │
│ BAL_004│  0.45   │  0.78   │  0.41   │  1.00   │  0.67     │
│ BAL_005│  0.12   │  0.56   │  0.19   │  0.67   │  1.00     │
│                                                             │
│ Leyenda: 🟢 Alta (>0.8) 🟡 Media (0.5-0.8) 🔴 Baja (<0.5) │
└─────────────────────────────────────────────────────────────┘
```

#### Análisis Detallado

Para cada comparación, el sistema proporciona:

**1. Puntuación de Similitud**
```
┌─── Análisis de Similitud ───┐
│                             │
│ Muestras: BAL_001 ↔ BAL_003 │
│                             │
│ Similitud Coseno: 0.87      │
│ Confianza: 94%              │
│ Clasificación: 🟢 ALTA      │
│                             │
│ Detalles:                   │
│ • Características: 256/256  │
│ • Correlación: 0.89         │
│ • Distancia: 0.13           │
│                             │
│ Recomendación:              │
│ ✅ Posible coincidencia     │
│    Revisar manualmente      │
└─────────────────────────────┘
```

**2. Visualización Gráfica**
- **Gráfico de barras**: Comparación de características
- **Mapa de calor**: Distribución de similitudes
- **Histograma**: Distribución de puntuaciones

**3. Estadísticas**
- **Media**: Similitud promedio
- **Desviación estándar**: Variabilidad
- **Percentiles**: Distribución de valores

---

## Configuración del Sistema

### Configuración General

#### Acceso a Configuración

1. **Vaya a la pestaña "⚙️ Configuración"**
2. **Seleccione la categoría deseada**
3. **Modifique los parámetros**
4. **Aplique los cambios**

#### Categorías de Configuración

```
┌─── Configuración del Sistema ───┐
│                                 │
│ 📁 General                      │
│ 🔧 Algoritmos                   │
│ 🗄️ Base de Datos               │
│ 🎨 Interfaz                     │
│ 🔒 Seguridad                    │
│ 📊 Rendimiento                  │
│ 📝 Logging                      │
│                                 │
└─────────────────────────────────┘
```

### Configuración de Algoritmos

#### Parámetros LBP

```
┌─── Configuración LBP ───────────┐
│                                 │
│ Radio:                          │
│ [2____] píxeles (1-5)           │
│                                 │
│ Número de Puntos:               │
│ [16___] puntos (8-24)           │
│                                 │
│ Método:                         │
│ ⚫ Uniforme                      │
│ ⚪ No uniforme                   │
│                                 │
│ Normalización:                  │
│ ☑ Normalizar histograma         │
│ ☑ Aplicar suavizado             │
│                                 │
│ [Aplicar] [Restablecer]         │
└─────────────────────────────────┘
```

#### Parámetros SIFT

```
┌─── Configuración SIFT ──────────┐
│                                 │
│ Número de Características:      │
│ [500___] (100-2000)             │
│                                 │
│ Umbral de Contraste:            │
│ [0.04__] (0.01-0.1)             │
│                                 │
│ Umbral de Borde:                │
│ [10____] (5-20)                 │
│                                 │
│ Sigma:                          │
│ [1.6___] (1.0-2.0)              │
│                                 │
│ [Aplicar] [Restablecer]         │
└─────────────────────────────────┘
```

### Configuración de Base de Datos

#### Conexión

```
┌─── Configuración de Base de Datos ──┐
│                                     │
│ Servidor:                           │
│ Host: [localhost____________]       │
│ Puerto: [5432]                      │
│                                     │
│ Base de Datos:                      │
│ Nombre: [sigec_balistica____]       │
│                                     │
│ Credenciales:                       │
│ Usuario: [sigec_user________]       │
│ Contraseña: [••••••••••••••]       │
│                                     │
│ Opciones:                           │
│ ☑ SSL habilitado                    │
│ ☑ Pool de conexiones               │
│ Pool size: [20__]                   │
│                                     │
│ [Probar Conexión] [Guardar]         │
└─────────────────────────────────────┘
```

#### Mantenimiento

- **Optimización automática**: Reorganizar índices
- **Limpieza de registros**: Eliminar datos temporales
- **Respaldo automático**: Configurar copias de seguridad
- **Estadísticas**: Monitorear rendimiento

### Configuración de Interfaz

#### Personalización

```
┌─── Configuración de Interfaz ───┐
│                                 │
│ Tema:                           │
│ ⚫ Claro                         │
│ ⚪ Oscuro                        │
│ ⚪ Automático                    │
│                                 │
│ Idioma:                         │
│ [Español ▼]                     │
│                                 │
│ Tamaño de Fuente:               │
│ [12__] puntos (8-24)            │
│                                 │
│ Opciones de Vista:              │
│ ☑ Mostrar barra de estado       │
│ ☑ Mostrar barra de herramientas │
│ ☑ Mostrar miniaturas            │
│                                 │
│ [Aplicar] [Restablecer]         │
└─────────────────────────────────┘
```

---

## Reportes y Resultados

### Tipos de Reportes

#### 1. Reporte de Análisis Individual

Información detallada de una muestra específica:

```
═══════════════════════════════════════════════════════════════
                    REPORTE DE ANÁLISIS
                     SIGeC-Balistica
═══════════════════════════════════════════════════════════════

INFORMACIÓN GENERAL
───────────────────
ID de Muestra:      BAL_2024_001
Nombre:             Muestra_Casquillo_001
Tipo:               Casquillo
Fecha de Análisis:  2024-01-15 14:30:25
Investigador:       Juan Pérez
Caso:               CASO_2024_001

INFORMACIÓN TÉCNICA
───────────────────
Algoritmo:          LBP (Local Binary Patterns)
Dimensiones:        256 características
Tiempo de Proceso:  2.3 segundos
Calidad de Imagen:  Alta (1920x1080)
Formato Original:   JPEG

CARACTERÍSTICAS EXTRAÍDAS
─────────────────────────
Histograma LBP:     [Ver gráfico adjunto]
Patrones Dominantes: 23, 45, 67, 89, 156
Uniformidad:        0.87
Contraste:          0.65
Entropía:           4.23

METADATOS
─────────
Archivo Original:   muestra_001.jpg
Tamaño:            2.3 MB
Fecha Creación:    2024-01-15 10:15:00
Cámara:            Canon EOS 5D Mark IV
Configuración:     ISO 100, f/8, 1/125s

═══════════════════════════════════════════════════════════════
Generado el: 2024-01-15 15:00:00
Sistema: SIGeC-Balistica v1.0
═══════════════════════════════════════════════════════════════
```

#### 2. Reporte de Comparación

Resultados de comparación entre muestras:

```
═══════════════════════════════════════════════════════════════
                   REPORTE DE COMPARACIÓN
                     SIGeC-Balistica
═══════════════════════════════════════════════════════════════

MUESTRAS COMPARADAS
───────────────────
Muestra A:          BAL_2024_001 (Casquillo)
Muestra B:          BAL_2024_003 (Casquillo)
Fecha Comparación:  2024-01-16 09:15:30
Analista:           María García

RESULTADOS DE SIMILITUD
───────────────────────
Similitud Coseno:   0.87 (87%)
Distancia Euclidiana: 0.13
Correlación:        0.89
Clasificación:      🟢 ALTA SIMILITUD

ANÁLISIS ESTADÍSTICO
────────────────────
Confianza:          94%
P-valor:            < 0.001
Significancia:      Estadísticamente significativa
Recomendación:      Revisar manualmente

CARACTERÍSTICAS COINCIDENTES
────────────────────────────
Patrones Comunes:   156 de 256 (61%)
Diferencias:        100 de 256 (39%)
Regiones Críticas:  Zona de percusión, Extractor

CONCLUSIÓN
──────────
Las muestras BAL_2024_001 y BAL_2024_003 presentan una alta
similitud (87%), sugiriendo una posible coincidencia balística.
Se recomienda análisis manual adicional para confirmación.

═══════════════════════════════════════════════════════════════
```

#### 3. Reporte de Lote

Análisis de múltiples muestras procesadas:

```
═══════════════════════════════════════════════════════════════
                     REPORTE DE LOTE
                     SIGeC-Balistica
═══════════════════════════════════════════════════════════════

INFORMACIÓN DEL LOTE
────────────────────
ID de Lote:         LOTE_2024_001
Fecha Proceso:      2024-01-16
Total Muestras:     25
Tiempo Total:       45 minutos
Analista:           Sistema Automático

RESUMEN DE PROCESAMIENTO
────────────────────────
✅ Exitosos:        23 (92%)
❌ Errores:         2 (8%)
⏳ Pendientes:      0 (0%)

ESTADÍSTICAS
────────────
Tiempo Promedio:    1.8 segundos/muestra
Algoritmo Usado:    LBP
Calidad Promedio:   Alta

MUESTRAS CON ERRORES
────────────────────
• BAL_2024_015: Error de formato de imagen
• BAL_2024_022: Imagen corrupta

DISTRIBUCIÓN POR TIPO
─────────────────────
Casquillos:         15 (60%)
Proyectiles:        10 (40%)

═══════════════════════════════════════════════════════════════
```

### Exportación de Reportes

#### Formatos Disponibles

**1. PDF**
- Formato profesional
- Incluye gráficos e imágenes
- Listo para impresión

**2. HTML**
- Visualización web
- Interactivo
- Fácil compartir

**3. CSV**
- Datos tabulares
- Compatible con Excel
- Análisis estadístico

**4. JSON**
- Formato estructurado
- Intercambio de datos
- Procesamiento automático

#### Proceso de Exportación

1. **Seleccione el reporte** a exportar
2. **Elija el formato** deseado
3. **Configure las opciones**:

```
┌─── Opciones de Exportación ─────┐
│                                 │
│ Formato: [PDF ▼]                │
│                                 │
│ Incluir:                        │
│ ☑ Imágenes originales           │
│ ☑ Gráficos de análisis          │
│ ☑ Metadatos completos           │
│ ☑ Estadísticas                  │
│                                 │
│ Calidad de Imagen:              │
│ ⚫ Alta (300 DPI)                │
│ ⚪ Media (150 DPI)               │
│ ⚪ Baja (72 DPI)                 │
│                                 │
│ [📤 Exportar] [❌ Cancelar]      │
└─────────────────────────────────┘
```

4. **Seleccione la ubicación** de guardado
5. **Confirme la exportación**

---

## Solución de Problemas

### Problemas Comunes

#### 1. Error al Cargar Imagen

**Síntomas**:
- Mensaje: "Error al cargar la imagen"
- La imagen no se muestra
- Proceso se detiene

**Soluciones**:
1. **Verificar formato**: Asegúrese de usar JPG, PNG, TIFF o BMP
2. **Verificar tamaño**: Máximo 50 MB por imagen
3. **Verificar integridad**: La imagen no debe estar corrupta
4. **Verificar permisos**: El archivo debe ser legible

```bash
# Verificar formato de imagen
file imagen.jpg

# Verificar integridad
identify imagen.jpg

# Verificar permisos
ls -la imagen.jpg
```

#### 2. Error de Conexión a Base de Datos

**Síntomas**:
- Mensaje: "No se puede conectar a la base de datos"
- Los registros no se cargan
- No se pueden guardar análisis

**Soluciones**:
1. **Verificar configuración**:
   - Host y puerto correctos
   - Credenciales válidas
   - Base de datos existe

2. **Verificar conectividad**:
```bash
# Probar conexión
telnet localhost 5432

# Verificar servicio PostgreSQL
systemctl status postgresql
```

3. **Verificar logs**:
```bash
# Logs de la aplicación
tail -f /var/log/sigec-balistica/app.log

# Logs de PostgreSQL
tail -f /var/log/postgresql/postgresql-*.log
```

#### 3. Procesamiento Lento

**Síntomas**:
- Análisis toma mucho tiempo
- Interfaz se congela
- Uso alto de CPU/memoria

**Soluciones**:
1. **Optimizar configuración**:
   - Reducir número de características
   - Usar algoritmos más rápidos (ORB vs SIFT)
   - Habilitar procesamiento paralelo

2. **Verificar recursos**:
```bash
# Monitorear recursos
htop
free -h
df -h
```

3. **Configurar límites**:
   - Tamaño máximo de imagen
   - Número máximo de muestras simultáneas
   - Timeout de procesamiento

#### 4. Interfaz No Responde

**Síntomas**:
- Ventanas no se actualizan
- Botones no funcionan
- Aplicación parece congelada

**Soluciones**:
1. **Esperar**: Algunos procesos pueden tomar tiempo
2. **Verificar logs**: Buscar errores en los registros
3. **Reiniciar aplicación**: Cerrar y abrir nuevamente
4. **Verificar recursos**: Memoria y CPU disponibles

### Códigos de Error

#### Errores de Sistema

| Código | Descripción | Solución |
|--------|-------------|----------|
| SYS_001 | Error de memoria insuficiente | Liberar memoria o aumentar RAM |
| SYS_002 | Error de espacio en disco | Liberar espacio o cambiar ubicación |
| SYS_003 | Error de permisos | Verificar permisos de archivos/directorios |

#### Errores de Base de Datos

| Código | Descripción | Solución |
|--------|-------------|----------|
| DB_001 | Conexión fallida | Verificar configuración de conexión |
| DB_002 | Consulta inválida | Reportar error al administrador |
| DB_003 | Registro duplicado | Verificar ID único |

#### Errores de Procesamiento

| Código | Descripción | Solución |
|--------|-------------|----------|
| PROC_001 | Imagen no válida | Verificar formato y integridad |
| PROC_002 | Algoritmo falló | Probar con otro algoritmo |
| PROC_003 | Timeout de proceso | Aumentar límite de tiempo |

### Logs y Diagnóstico

#### Ubicación de Logs

**Aplicación**:
- `/var/log/sigec-balistica/app.log`
- `/var/log/sigec-balistica/error.log`

**Sistema**:
- `/var/log/syslog`
- `journalctl -u sigec-balistica`

#### Niveles de Log

- **DEBUG**: Información detallada para desarrollo
- **INFO**: Información general de funcionamiento
- **WARNING**: Advertencias que no impiden funcionamiento
- **ERROR**: Errores que afectan funcionalidad
- **CRITICAL**: Errores críticos que detienen el sistema

#### Habilitar Debug

Para obtener más información de diagnóstico:

1. **Edite la configuración**:
```json
{
  "logging": {
    "level": "DEBUG",
    "enable_console": true
  }
}
```

2. **Reinicie la aplicación**
3. **Reproduzca el problema**
4. **Revise los logs detallados**

---

## Preguntas Frecuentes

### Generales

**P: ¿Qué formatos de imagen soporta el sistema?**
R: JPG, PNG, TIFF y BMP. Se recomienda usar JPG o PNG para mejor rendimiento.

**P: ¿Cuál es el tamaño máximo de imagen?**
R: 50 MB por imagen. Para imágenes más grandes, reduzca la resolución manteniendo la calidad.

**P: ¿Puedo procesar múltiples imágenes simultáneamente?**
R: Sí, el sistema soporta procesamiento por lotes de hasta 100 imágenes.

### Técnicas

**P: ¿Qué algoritmo es mejor para mi caso?**
R: 
- **LBP**: Rápido, bueno para texturas y patrones
- **SIFT**: Preciso, invariante a escala y rotación
- **ORB**: Equilibrio entre velocidad y precisión

**P: ¿Cómo interpreto los valores de similitud?**
R:
- **> 0.8**: Alta similitud, posible coincidencia
- **0.5 - 0.8**: Similitud media, revisar manualmente
- **< 0.5**: Baja similitud, probablemente diferentes

**P: ¿Puedo comparar diferentes tipos de muestras?**
R: Técnicamente sí, pero se recomienda comparar solo muestras del mismo tipo (casquillo con casquillo, proyectil con proyectil).

### Rendimiento

**P: ¿Por qué el análisis es lento?**
R: Factores que afectan velocidad:
- Tamaño de imagen
- Algoritmo seleccionado
- Recursos del sistema
- Configuración de parámetros

**P: ¿Cómo puedo acelerar el procesamiento?**
R:
- Use imágenes de menor resolución
- Seleccione ORB en lugar de SIFT
- Habilite procesamiento paralelo
- Aumente la memoria RAM

### Base de Datos

**P: ¿Cuántas muestras puede almacenar el sistema?**
R: No hay límite teórico. En la práctica, depende del espacio en disco y rendimiento deseado.

**P: ¿Cómo hago respaldo de mis datos?**
R: Use la función de exportación o configure respaldos automáticos en la configuración del sistema.

**P: ¿Puedo migrar datos de otro sistema?**
R: Sí, a través de la función de importación. Contacte al soporte para formatos específicos.

### Seguridad

**P: ¿Los datos están seguros?**
R: Sí, el sistema incluye:
- Cifrado de datos sensibles
- Autenticación de usuarios
- Logs de auditoría
- Respaldos automáticos

**P: ¿Puedo controlar el acceso de usuarios?**
R: Sí, a través del sistema de roles y permisos (requiere configuración de administrador).

### Soporte

**P: ¿Dónde puedo obtener ayuda adicional?**
R: 
- **Documentación**: Manual técnico completo
- **Email**: soporte@sigec-balistica.com
- **Teléfono**: +1-800-SIGEC-HELP
- **Web**: https://support.sigec-balistica.com

**P: ¿Hay actualizaciones disponibles?**
R: Las actualizaciones se notifican automáticamente. También puede verificar en "Ayuda > Buscar Actualizaciones".

**P: ¿Ofrecen capacitación?**
R: Sí, ofrecemos:
- Capacitación en línea
- Sesiones presenciales
- Webinars mensuales
- Documentación interactiva

---

## Apéndices

### A. Atajos de Teclado

| Acción | Atajo | Descripción |
|--------|-------|-------------|
| Abrir archivo | Ctrl+O | Cargar nueva imagen |
| Guardar | Ctrl+S | Guardar análisis actual |
| Nuevo análisis | Ctrl+N | Iniciar nuevo análisis |
| Comparar | Ctrl+C | Abrir herramienta de comparación |
| Configuración | Ctrl+, | Abrir configuración |
| Ayuda | F1 | Mostrar ayuda |
| Pantalla completa | F11 | Alternar pantalla completa |
| Salir | Ctrl+Q | Cerrar aplicación |

### B. Especificaciones Técnicas

**Algoritmos Soportados**:
- LBP (Local Binary Patterns)
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)

**Métricas de Similitud**:
- Similitud Coseno
- Distancia Euclidiana
- Correlación de Pearson
- Distancia de Manhattan

**Formatos de Exportación**:
- PDF (Portable Document Format)
- HTML (HyperText Markup Language)
- CSV (Comma-Separated Values)
- JSON (JavaScript Object Notation)
- XML (eXtensible Markup Language)

### C. Glosario

**Algoritmo**: Conjunto de reglas o instrucciones para resolver un problema específico.

**Características**: Propiedades distintivas extraídas de una imagen para análisis.

**Histograma**: Representación gráfica de la distribución de valores en un conjunto de datos.

**LBP**: Operador de textura que describe patrones locales en una imagen.

**Métrica de Similitud**: Medida matemática para cuantificar la similitud entre dos elementos.

**NIST**: Instituto Nacional de Estándares y Tecnología de Estados Unidos.

**Similitud Coseno**: Medida de similitud basada en el ángulo entre dos vectores.

**SIFT**: Algoritmo para detectar y describir características locales en imágenes.

---

*Manual de Usuario - SIGeC-Balistica v1.0*  
*Última actualización: Enero 2024*  
*© 2024 SIGeC-Balistica. Todos los derechos reservados.*