# SIGeC-Balisticar v0.1.3 - Sistema Integrado de Gestión y Control Balístico

Sistema integrado para el análisis forense automatizado de cartuchos y balas utilizando técnicas avanzadas de visión por computadora, aprendizaje automático y deep learning, conforme a estándares NIST y AFTE.

## Descripción

SIGeC-Balisticar es una herramienta avanzada de análisis balístico forense que permite:

- Extracción automática de características de cartuchos y balas
- Comparación y matching de evidencia balística con algoritmos CMC
- Análisis estadístico de patrones de marcas conforme a NIST
- Interfaz gráfica PyQt5 para análisis forense
- Integración con bases de datos balísticas vectoriales
- Pipeline científico unificado para análisis completo

## Arquitectura del Sistema

```
SIGeC-Balisticar/
├── 📁 assets/                  # Imágenes y recursos de prueba
├── 📁 common/                  # Núcleo estadístico y adaptadores NIST
├── 📁 config/                  # Configuración unificada (YAML/gestor)
├── 📁 core/                    # Pipeline científico y sistemas centrales
├── 📁 database/                # Base de datos unificada y vectorial
├── 📁 deep_learning/           # Estructura para modelos y utilidades
├── 📁 gui/                     # Interfaz gráfica PyQt5
├── 📁 image_processing/        # Lazy loading y cache LBP
├── 📁 matching/                # Algoritmos de matching y CMC
├── 📁 nist_standards/          # Esquemas y validación NIST
├── 📁 performance/             # Monitoreo y métricas
├── 📁 tests/                   # Suite de pruebas consolidada
├── 📁 utils/                   # Utilidades y validadores
├── 📁 DOCS/                    # Documentación del proyecto
├── 📄 main.py                  # Punto de entrada CLI/GUI
├── 📄 launch_gui.py            # Lanzador GUI
├── 📄 production_deployment.py # Utilidades de despliegue
├── 📄 pytest.ini               # Configuración de pytest
└── 📄 requirements.txt         # Dependencias
```

## Instalación Rápida

### Prerrequisitos
- Python 3.8+
- OpenCV 4.x
- PyQt5
- CUDA (opcional, para aceleración GPU)

### Pasos de Instalación

1. Clonar el repositorio
```bash
git clone <repository-url>
cd SIGeC-Balisticar
```

2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. Instalar dependencias
```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación GUI
```bash
python launch_gui.py
```

## Configuración

El sistema usa una configuración unificada basada en YAML con helpers Python.

- Archivos: `config/unified_config.yaml` (+ variantes por entorno)
- Módulo: `config/unified_config.py`

### Uso de la Configuración

```python
from config.unified_config import get_unified_config

# Cargar configuración por entorno
base = get_unified_config(env="base")
testing = get_unified_config(env="testing")
production = get_unified_config(env="production")

# Acceso a valores
db_host = base.get("database.host", default="localhost")
gui_theme = base.get("gui.theme", default="modern")
```

### Variables de Entorno (ejemplos)

```bash
export SIGEC_DATABASE_HOST="custom-db-server"
export SIGEC_GUI_THEME="dark"
export SIGEC_ENVIRONMENT="production"
```

## Uso del Sistema

### Interfaz Gráfica
```bash
python launch_gui.py
```

### Pipeline Científico (CLI)
```bash
python -m core.unified_pipeline imagen1.jpg imagen2.jpg --level forensic
```

### Análisis por Lotes
```bash
python scripts/batch_analysis.py --input_dir /path/to/images --output_dir /path/to/results
```

## Testing

### Ejecutar todas las pruebas
```bash
pytest tests/ -v
```

### Consejos útiles
- Si aparece `ImportError` en tests, exportar `PYTHONPATH=.` o añadir `pythonpath = .` en `pytest.ini`.
- Para pruebas de GUI en CI/headless: `QT_QPA_PLATFORM=offscreen`.

### Pruebas específicas
```bash
# Integración básica
pytest tests/test_basic_integration.py -v

# GUI headless consolidada
QT_QPA_PLATFORM=offscreen pytest tests/integration/test_frontend_integration_consolidated.py -q

# Rendimiento de caché
pytest tests/test_cache_performance.py -v
```

## Estado del Proyecto

### Módulos Principales
- Core, Matching, Image Processing, GUI, Database y Utils operativos
- Configuración unificada activa y en proceso de consolidación de usos legacy
- Suite de tests consolidada bajo `tests/` (unitarios, integración, GUI)

### Métricas de Calidad
- Cobertura y conteo de tests se están recalculando tras la consolidación

## Contribución

1. Fork del proyecto
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Licencia

Proyecto bajo Licencia MIT. Ver `LICENSE`.

## Soporte

- Crear un issue en GitHub
- Consultar documentación en `/DOCS/`
- Revisar logs del sistema en `/logs/`

## Referencias
- Pendiente

---

SIGeC-Balisticar v0.1.3 — Sistema Integral de Gestión Criminalística Argentino (Extensión: Análisis Balístico)