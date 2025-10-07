# Solución de Problemas Qt - SIGeC-Balistica

## Problema Identificado

Durante el desarrollo se identificó un problema crítico con la carga de plugins de Qt en el entorno de ejecución. Los síntomas incluían:

- Plugins Qt encontrados pero no cargables (`xcb`, `minimal`, `offscreen`, `wayland`)
- Crashes con código de salida 134 (SIGABRT)
- Core dumps al intentar crear aplicaciones Qt
- Mensajes de error: "This application failed to start because no Qt platform plugin could be initialized"

## Causa Raíz

El problema se debe a incompatibilidades entre:
- Versiones de Qt/PyQt5
- Dependencias de bibliotecas compartidas
- Configuración del sistema gráfico
- Permisos o bibliotecas gráficas faltantes

## Solución Implementada

### 1. Verificación Robusta de Qt (`is_qt_functional`)

Se implementó una función que verifica la funcionalidad real de Qt usando un proceso separado:

```python
def is_qt_functional():
    """
    Verifica si Qt está realmente funcional usando un proceso separado
    para evitar crashes del proceso principal.
    """
    import subprocess
    import tempfile
    
    # Crear script de prueba temporal que intenta crear QApplication
    test_script = '''
import sys
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'minimal'

try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    app = QApplication([])
    app.setApplicationName("Test")
    app.quit()
    print("SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
    
    # Ejecutar en proceso separado con timeout
    result = subprocess.run([sys.executable, test_script_path], 
                          capture_output=True, text=True, timeout=5)
    
    return result.returncode == 0 and "SUCCESS" in result.stdout
```

### 2. Fallback Automático a Modo Headless

La aplicación detecta automáticamente cuando Qt no está funcional y activa el modo headless:

```python
def create_application():
    """Crea la aplicación Qt o retorna None si no es posible"""
    if not is_qt_functional():
        logger.warning("Qt no está funcional - se usará modo headless")
        return None
    
    # Crear aplicación Qt solo si es funcional
    # ...
```

### 3. Modo Headless Completo

El modo headless proporciona toda la funcionalidad del backend sin interfaz gráfica:

- Sistema de configuración
- Procesamiento de imágenes
- Algoritmos de matching
- Base de datos vectorial
- API REST (si se implementa)

## Beneficios de la Solución

1. **Robustez**: La aplicación nunca crashea por problemas de Qt
2. **Detección Temprana**: Identifica problemas antes de intentar crear la GUI
3. **Fallback Automático**: Transición transparente a modo headless
4. **Funcionalidad Completa**: Todas las capacidades disponibles sin GUI
5. **Debugging Mejorado**: Logs claros sobre el estado de Qt

## Uso en Diferentes Entornos

### Entorno con GUI Funcional
```bash
python main.py --debug  # Inicia con GUI
```

### Entorno sin GUI (Servidores, Contenedores)
```bash
python main.py --headless  # Modo headless explícito
python main.py --debug     # Detecta automáticamente y usa headless
```

### Modo de Pruebas
```bash
python main.py --test      # Ejecuta tests (funciona en ambos modos)
```

## Verificación del Estado

La aplicación reporta su estado al iniciar:

```
2025-10-05 08:59:52 | INFO | Estado del sistema: {
    'config_available': True, 
    'statistical_available': True, 
    'nist_available': True, 
    'image_processing_available': True, 
    'matching_available': True, 
    'database_available': True, 
    'current_status': 'idle', 
    'backend_version': '1.0.0'
}
```

## Mantenimiento

- La verificación de Qt se ejecuta solo una vez al inicio
- Los logs proporcionan información detallada sobre el estado
- El fallback es completamente automático y transparente
- No requiere configuración manual del usuario

Esta solución garantiza que SIGeC-Balistica funcione correctamente en cualquier entorno, con o sin capacidades gráficas.