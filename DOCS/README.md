# SEACABAr - Sistema Experto de Análisis de Cartuchos y Balas Automático

Sistema experto para el análisis forense automatizado de cartuchos y balas utilizando técnicas de visión por computadora, aprendizaje automático y deep learning.

## Descripción

SEACABAr es una herramienta avanzada de análisis balístico forense que permite:

- Extracción automática de características de cartuchos y balas
- Comparación y matching de evidencia balística
- Análisis estadístico de patrones de marcas
- Interfaz gráfica intuitiva para análisis forense
- Integración con bases de datos balísticas

## Estructura del Proyecto

```
SEACABAr/
├── assets/                     # Recursos e imágenes de prueba
├── database/                   # Gestión de base de datos y vectores
├── deep_learning/              # Modelos de deep learning
├── docs/                       # Documentación completa
├── gui/                        # Interfaz gráfica de usuario
├── image_processing/           # Procesamiento de imágenes y extracción de características
├── matching/                   # Algoritmos de comparación y matching
├── scripts/                    # Scripts de utilidad
├── tests/                      # Pruebas unitarias e integración
├── utils/                      # Utilidades generales
├── config.yaml                 # Configuración principal
├── main.py                     # Punto de entrada de la aplicación
└── requirements.txt            # Dependencias del proyecto
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd SEACABAr
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate     # En Windows
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Interfaz Gráfica
```bash
python main.py
```

### Servicios Backend
```bash
# Extractor de características
python image_processing/feature_extractor.py

# Detector de ROI
python image_processing/roi_detector.py

# Analizador estadístico
python image_processing/statistical_analyzer.py
```

## Documentación

La documentación completa se encuentra en el directorio `docs/`:

- [Guía de Instalación](docs/GUIA_INSTALACION.md)
- [Guía de Usuario](docs/GUIA_USUARIO.md)
- [Arquitectura Técnica](docs/reports/ARQUITECTURA_TECNICA.md)
- [Reporte de Validación](docs/reports/VALIDATION_REPORT.md)

## Características Principales

- **Extracción de Características**: SIFT, ORB, LBP, características balísticas
- **Deep Learning**: Modelos CNN, Siamese Networks, Segmentación
- **Análisis Estadístico**: Métricas avanzadas y visualizaciones
- **Base de Datos**: SQLite + FAISS para búsquedas vectoriales
- **Interfaz Moderna**: PyQt6 con visualizaciones interactivas

## Contribución

Para contribuir al proyecto, consulte la documentación en `docs/` y siga las mejores prácticas establecidas.

## Licencia

Ver archivo `docs/reports/LICENCIAS.md` para información sobre licencias.