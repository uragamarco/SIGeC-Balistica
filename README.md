# SIGeC-Balisticar v0.1.3 - Sistema Integrado de Gestión y Control Balístico

Sistema integrado para el análisis forense automatizado de cartuchos y balas utilizando técnicas avanzadas de visión por computadora, aprendizaje automático y deep learning, conforme a estándares NIST y AFTE.

## Descripción

SIGeC-Balisticar es una herramienta avanzada de análisis balístico forense que permite:

- ✅ Extracción automática de características de cartuchos y balas
- ✅ Comparación y matching de evidencia balística con algoritmos CMC
- ✅ Análisis estadístico de patrones de marcas conforme a NIST
- ✅ Interfaz gráfica intuitiva para análisis forense
- ✅ Integración con bases de datos balísticas vectoriales
- ✅ Pipeline científico unificado para análisis completo
- ✅ Conclusiones AFTE automatizadas

## Arquitectura del Sistema

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
├── 📄 config.yaml              # Configuración principal
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

## Configuración

El sistema utiliza configuración unificada en `config.yaml`:

```yaml
database:
  type: "unified"
  path: "data/ballistic_db.db"

gui:
  theme: "modern"
  enable_gpu: true

image_processing:
  roi_detection: "watershed"
  feature_extraction: "orb_sift_hybrid"

matching:
  algorithm: "unified_matcher"
  cmc_threshold: 8
```

## Uso del Sistema

### Interfaz Gráfica
```bash
python main.py
```

### Pipeline Científico (CLI)
```bash
python -m core.unified_pipeline imagen1.jpg imagen2.jpg --level forensic
```

### Análisis por Lotes
```bash
python scripts/batch_analysis.py --input_dir /path/to/images --output_dir /path/to/results
```

## Características Principales

### 🔬 Pipeline Científico
- **Preprocesamiento NIST**: Normalización y mejora de calidad
- **Detección ROI**: Algoritmo Watershed optimizado
- **Extracción de Características**: ORB/SIFT híbrido
- **Matching Avanzado**: Algoritmo CMC con ponderación de calidad
- **Conclusiones AFTE**: Automatizadas y validadas

### Interfaz Gráfica
- **Visualización Interactiva**: Mapas de calor y correlaciones
- **Análisis en Tiempo Real**: Procesamiento asíncrono
- **Reportes Automáticos**: Exportación PDF/HTML
- **Base de Datos Integrada**: Gestión de casos y evidencia

### Rendimiento
- **Aceleración GPU**: CUDA y OpenCL
- **Procesamiento Paralelo**: Multi-threading optimizado
- **Cache Inteligente**: LBP y características pre-calculadas
- **Memoria Optimizada**: Gestión eficiente de recursos

## Testing

### Ejecutar todas las pruebas:
```bash
pytest tests/ -v
```

### Pruebas específicas:
```bash
# Pruebas de integración
pytest tests/test_basic_integration.py -v

# Pruebas de GUI (headless)
pytest tests/integration/test_gui_headless.py -v

# Benchmarks de rendimiento
pytest tests/test_performance_benchmarks.py -v
```

## Estado del Proyecto

**Estado Actual**: 

### Módulos Completados:
- ✅ **GUI**: Interfaz completa con todas las funcionalidades
- ✅ **Core**: Pipeline científico unificado
- ✅ **Image Processing**: Procesamiento avanzado y extracción de características
- ✅ **Matching**: Algoritmos CMC y matching unificado
- ✅ **Database**: Sistema de base de datos vectorial
- ✅ **NIST Standards**: Cumplimiento de estándares forenses
- ✅ **Testing**: Suite completa de pruebas

### Métricas de Calidad:
- **Cobertura de Código**: >85%
- **Pruebas Unitarias**: 45+ tests
- **Pruebas de Integración**: 15+ tests
- **Documentación**: Completa y actualizada

## Contribución

1. Fork del proyecto
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Soporte

Para soporte técnico o reportar bugs:
- Crear un issue en GitHub
- Consultar la documentación en `/DOCS/`
- Revisar los logs del sistema en `/logs/`

## Referencias

- *Pendiente

---

**SIGeC-Balistica v0.1.3** - Sistema Integral de Gestion Criminalístico Argentino - Extensión: Análisis Balístico 