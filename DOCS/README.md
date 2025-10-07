# SIGeC-Balisticar - Documentación Técnica

## Índice de Documentación

Esta carpeta contiene toda la documentación técnica del proyecto SIGeC-Balisticar.

### 📋 Documentación Principal

- **[README.md](../README.md)** - Documentación principal del proyecto
- **[INFORME_ESTADO_PROYECTO.md](INFORME_ESTADO_PROYECTO.md)** - Estado actual del proyecto
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Plan de implementación
- **[ESTADO_PLAN_DESARROLLO.md](ESTADO_PLAN_DESARROLLO.md)** - Estado del plan de desarrollo

### 🔧 Análisis Técnico por Módulos

- **[analisis_common.md](analisis_common.md)** - Análisis del módulo common
- **[analisis_core.md](analisis_core.md)** - Análisis del módulo core
- **[analisis_database.md](analisis_database.md)** - Análisis del módulo database
- **[analisis_deep_learning.md](analisis_deep_learning.md)** - Análisis del módulo deep_learning
- **[analisis_image_processing.md](analisis_image_processing.md)** - Análisis del módulo image_processing
- **[analisis_matching.md](analisis_matching.md)** - Análisis del módulo matching
- **[analisis_nist_standards.md](analisis_nist_standards.md)** - Análisis del módulo nist_standards
- **[analisis_performance.md](analisis_performance.md)** - Análisis del módulo performance
- **[analisis_utils.md](analisis_utils.md)** - Análisis del módulo utils

### 🚀 Despliegue y Optimización

- **[deployment_summary.md](deployment_summary.md)** - Resumen de despliegue
- **[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md)** - Reporte de optimización
- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - Resumen de optimización
- **[RESUMEN_LIMPIEZA_REPOSITORIO.md](RESUMEN_LIMPIEZA_REPOSITORIO.md)** - Resumen de limpieza del repositorio

### 🛠️ Solución de Problemas

- **[qt_troubleshooting.md](qt_troubleshooting.md)** - Solución de problemas con PyQt5

### 📊 Herramientas de Documentación

- **[documentation_system.py](documentation_system.py)** - Sistema automatizado de documentación

## Estructura del Proyecto

```
SIGeC-Balisticar/
├── 📁 assets/                  # Recursos e imágenes de prueba
├── 📁 common/                  # Núcleo estadístico y adaptadores NIST
├── 📁 config/                  # Configuraciones unificadas
├── 📁 core/                    # Pipeline científico y sistemas centrales
├── 📁 database/                # Base de datos unificada y vectorial
├── 📁 deep_learning/           # Modelos CNN y Siameses
├── 📁 gui/                     # Interfaz gráfica PyQt5
├── 📁 image_processing/        # Procesamiento avanzado de imágenes
├── 📁 matching/                # Algoritmos de matching y CMC
├── 📁 nist_standards/          # Estándares NIST y validación
├── 📁 performance/             # Monitoreo y optimización
├── 📁 tests/                   # Suite de pruebas completa
├── 📁 utils/                   # Utilidades y validadores
├── 📄 main.py                  # Punto de entrada
└── 📄 requirements.txt         # Dependencias
```

## Instalación Rápida

### Prerrequisitos
- Python 3.8+
- OpenCV 4.x
- PyQt5
- CUDA (opcional, para aceleración GPU)

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd SIGeC-Balisticar
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación:**
```bash
python main.py
```

## Contribución

Para contribuir al proyecto, consulte la documentación técnica específica de cada módulo y siga las guías de desarrollo establecidas.

## Soporte

Para problemas técnicos, consulte primero la documentación de solución de problemas y los análisis técnicos por módulos.