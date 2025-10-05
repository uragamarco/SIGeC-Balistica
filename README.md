# SIGeC-Balistica v2.0.0

**Sistema Integrado de Gestión y Control Balístico**

## Descripción

SIGeC-Balistica es un sistema avanzado de análisis forense balístico que utiliza técnicas de visión por computadora, aprendizaje automático y análisis estadístico para la identificación y comparación automatizada de cartuchos de armas de fuego. El sistema está diseñado para asistir a peritos forenses en la evaluación de evidencia balística.

## Características Principales

### Análisis de Imágenes
- **Procesamiento avanzado de imágenes**: Algoritmos optimizados para análisis de cartuchos
- **Detección automática de ROI**: Identificación inteligente de regiones de interés
- **Múltiples algoritmos**: ORB, SIFT, LBP y técnicas híbridas
- **Soporte multi-formato**: JPG, PNG, TIFF, BMP

### Matching y Comparación
- **Algoritmos unificados**: Sistema de matching híbrido optimizado
- **Análisis CMC**: Implementación de Cumulative Match Characteristic
- **Matching paralelo**: Procesamiento multi-hilo para mejor rendimiento
- **Validación estadística**: Análisis de confiabilidad y precisión

### Base de Datos
- **SQLite optimizado**: Base de datos local de alto rendimiento
- **Índices FAISS**: Búsqueda vectorial ultra-rápida
- **Backup automático**: Sistema de respaldo y recuperación
- **Escalabilidad**: Manejo eficiente de grandes volúmenes de datos

### Análisis Estadístico
- **Integración NIST**: Compatibilidad con estándares NIST
- **Visualizaciones avanzadas**: Gráficos interactivos y reportes
- **Métricas forenses**: Análisis de calidad y confiabilidad
- **Exportación de reportes**: Formatos PDF, HTML, Excel

### Interfaz Gráfica
- **PyQt5**: Interfaz moderna y responsiva
- **Diseño intuitivo**: Flujo de trabajo optimizado para peritos
- **Visualización en tiempo real**: Resultados inmediatos
- **Configuración flexible**: Parámetros ajustables por el usuario

## Instalación

### Requisitos del Sistema
- **Python**: 3.8 o superior
- **RAM**: 8GB recomendado (mínimo 4GB)
- **Espacio en disco**: 2GB libres
- **SO**: Windows 10/11, Linux Ubuntu 18.04+

### Dependencias Principales
```bash
pip install -r requirements.txt
```

Principales librerías:
- PyQt5 >= 5.15.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- scikit-image >= 0.18.0
- FAISS-CPU >= 1.7.0
- Matplotlib >= 3.5.0

### Configuración
1. Clonar el repositorio
2. Instalar dependencias
3. Configurar `config.yaml` según necesidades
4. Ejecutar `python main.py`

## Estructura del Proyecto

```
SIGeC-Balistica/
├── api/                    # API y sistema de optimización
├── assets/                 # Recursos gráficos
├── common/                 # Módulos compartidos
├── config/                 # Configuraciones
├── core/                   # Núcleo del sistema
├── database/               # Gestión de base de datos
├── deep_learning/          # Módulos de ML/DL
├── gui/                    # Interfaz gráfica
├── image_processing/       # Procesamiento de imágenes
├── matching/               # Algoritmos de matching
├── nist_standards/         # Integración NIST
├── tests/                  # Suite de pruebas
├── utils/                  # Utilidades
└── main.py                 # Punto de entrada
```

## Uso

### Inicio Rápido
```bash
python main.py
```

### Análisis Básico
1. **Cargar imágenes**: Importar cartuchos desde archivos
2. **Configurar parámetros**: Ajustar algoritmos y umbrales
3. **Ejecutar análisis**: Procesar y obtener resultados
4. **Revisar matches**: Evaluar coincidencias encontradas
5. **Generar reporte**: Exportar resultados

### Configuración Avanzada
El archivo `config.yaml` permite personalizar:
- Parámetros de algoritmos
- Configuración de base de datos
- Opciones de interfaz
- Niveles de logging

## Rendimiento

### Benchmarks
- **Procesamiento**: ~2-5 segundos por imagen (CPU)
- **Matching**: ~100-500ms por comparación
- **Base de datos**: Soporte para 100K+ registros
- **Memoria**: Uso optimizado < 2GB RAM

### Optimizaciones
- Cache inteligente de características
- Procesamiento paralelo
- Índices vectoriales FAISS
- Gestión eficiente de memoria

## Testing

```bash
# Ejecutar todas las pruebas
pytest tests/

# Pruebas específicas
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Cobertura de Pruebas
- Pruebas unitarias: Módulos individuales
- Pruebas de integración: Flujos completos
- Pruebas de rendimiento: Benchmarks
- Pruebas de GUI: Interfaz de usuario

## Validación Científica

### Estándares NIST
- Compatibilidad con NIST Ballistics Toolmark Database
- Implementación de métricas estándar
- Validación con datasets públicos
- Trazabilidad de resultados

### Métricas de Evaluación
- **Precisión**: Tasa de verdaderos positivos
- **Recall**: Sensibilidad del sistema
- **F1-Score**: Medida armónica
- **CMC**: Curvas de matching acumulativo

## Seguridad

- Manejo seguro de datos sensibles
- Logging auditado
- Validación de entrada
- Gestión de errores robusta

## Documentación

### Documentos Técnicos
- `DOCS/README.md`: Documentación completa
- `DOCS/OPTIMIZATION_REPORT.md`: Análisis de rendimiento
- `DOCS/deployment_summary.md`: Guía de despliegue

### API Reference
Documentación completa de APIs disponible en el código fuente con docstrings detallados.

## Contribución

### Desarrollo
1. Fork del repositorio
2. Crear rama feature
3. Implementar cambios
4. Ejecutar pruebas
5. Crear Pull Request

### Estándares de Código
- PEP 8 para Python
- Docstrings obligatorios
- Type hints recomendados
- Cobertura de pruebas > 80%

## Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

## Autores

- **Himo "Larry" Anon** - Desarrollo principal - marcouraga.1992@gmail.com

## Agradecimientos

- Comunidad NIST por estándares y datasets
- Contribuidores de OpenCV y scikit-image
- Equipo de desarrollo de PyQt5
- Comunidad forense por feedback y validación

## Soporte

Para soporte técnico o consultas:
- **Email**: marcouraga.1992@gmail.com
- **Issues**: GitHub Issues
- **Documentación**: Ver carpeta `DOCS/`

---

**SIGeC-Balistica v0.1.3** - Sistema Integrado de Gestión y Control Balístico