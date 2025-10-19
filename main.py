#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIGeC-Balistica - Sistema Experto de Análisis de Cartuchos y Balas Automático
=====================================================================

Aplicación principal que integra todas las funcionalidades del sistema forense balístico:
- Análisis individual de cartuchos, balas y proyectiles
- Comparación directa y búsqueda en base de datos balística
- Gestión de base de datos de evidencia balística
- Generación de reportes forenses con estándares NIST y conclusiones AFTE
- Sistema de monitoreo y métricas avanzado

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
Versión: 2.0.0
"""

import sys
import os
import logging
import argparse
import traceback
import atexit
from pathlib import Path

# Añadir el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar sistema de logging centralizado
from utils.logger import get_logger

# Logger inicial (se configurará en main())
logger = None

# Importar sistema de monitoreo
try:
    from monitoring.integration import (
        initialize_monitoring, 
        start_monitoring, 
        stop_monitoring,
        get_system_metrics
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    # Funciones fallback
    def initialize_monitoring(config=None): return True
    def start_monitoring(): return True
    def stop_monitoring(): pass
    def get_system_metrics(): return {}

def check_dependencies():
    """Verifica que todas las dependencias estén disponibles"""
    
    logger = get_logger("dependencies")
    
    missing_deps = []
    
    # Dependencias críticas
    critical_deps = [
        ('PyQt5', 'PyQt5.QtWidgets'),
        ('numpy', 'numpy'),
        ('opencv', 'cv2'),
        ('PIL', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('pandas', 'pandas'),
    ]
    
    # Dependencias opcionales
    optional_deps = [
        ('psutil', 'psutil'),
        ('reportlab', 'reportlab'),
        ('jinja2', 'jinja2'),
        ('yaml', 'yaml'),
    ]
    
    logger.info("Verificando dependencias...")
    
    # Verificar dependencias críticas
    for name, module in critical_deps:
        try:
            __import__(module)
            logger.debug(f"✓ {name} disponible")
        except ImportError:
            missing_deps.append(name)
            logger.error(f"✗ {name} no disponible")
    
    # Verificar dependencias opcionales
    for name, module in optional_deps:
        try:
            __import__(module)
            logger.debug(f"✓ {name} disponible (opcional)")
        except ImportError:
            logger.warning(f"⚠ {name} no disponible (opcional)")
    
    if missing_deps:
        logger.error(f"Dependencias críticas faltantes: {', '.join(missing_deps)}")
        logger.error("Instale las dependencias con: pip install -r requirements.txt")
        return False
    
    logger.info("✓ Todas las dependencias críticas están disponibles")
    return True

def check_system_requirements():
    """Verifica los requisitos del sistema"""
    
    logger = get_logger("system_requirements")
    
    logger.info("Verificando requisitos del sistema...")
    
    # Verificar Python
    python_version = sys.version_info
    if python_version < (3, 7):
        logger.error(f"Python 3.7+ requerido, encontrado {python_version.major}.{python_version.minor}")
        return False
    
    logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar memoria disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb < 4:
            logger.warning(f"Memoria RAM baja: {memory_gb:.1f}GB (recomendado: 8GB+)")
        else:
            logger.info(f"✓ Memoria RAM: {memory_gb:.1f}GB")
            
    except ImportError:
        logger.warning("No se pudo verificar la memoria (psutil no disponible)")
    
    # Verificar espacio en disco
    try:
        import shutil
        disk_usage = shutil.disk_usage(project_root)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 1:
            logger.warning(f"Espacio en disco bajo: {free_gb:.1f}GB")
        else:
            logger.info(f"✓ Espacio libre: {free_gb:.1f}GB")
            
    except Exception as e:
        logger.warning(f"No se pudo verificar el espacio en disco: {e}")
    
    return True

def setup_environment():
    """Configura el entorno de la aplicación"""
    
    logger = get_logger("environment")
    
    logger.info("Configurando entorno...")
    
    try:
        # Crear directorios necesarios
        directories = [
            "data",
            "data/images",
            "data/database", 
            "data/exports",
            "data/reports",
            "data/temp",
            "logs",
            "config"
        ]
        
        for dir_name in directories:
            dir_path = project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"✓ Directorio: {dir_path}")
        
        # Configurar variables de entorno
        logger.debug("Configurando variables de entorno...")
        os.environ['SIGEC_BALISTICA_ROOT'] = str(project_root)
        os.environ['SIGEC_BALISTICA_DATA'] = str(project_root / "data")
        logger.debug("✓ Variables de entorno configuradas")
        
        logger.info("✓ Entorno configurado")
        
    except Exception as e:
        logger.error(f"Error configurando entorno: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise  # Re-lanzar la excepción para que sea manejada por el caller

def is_qt_functional():
    """
    Verifica si Qt está realmente funcional
    """
    logger = get_logger("qt_check")
    logger.info("=== VERIFICANDO FUNCIONALIDAD DE QT ===")
    
    # Obtener la ruta de plugins de PyQt5 dinámicamente
    try:
        logger.debug("Paso 1: Intentando importar PyQt5...")
        import PyQt5
        pyqt5_path = os.path.dirname(PyQt5.__file__)
        plugin_dir = os.path.join(pyqt5_path, 'Qt5', 'plugins')
        logger.debug(f"✓ PyQt5 importado desde: {pyqt5_path}")
        logger.debug(f"✓ Plugin directory esperado: {plugin_dir}")
        
        if not os.path.exists(plugin_dir):
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            # Intentar rutas alternativas
            alt_plugin_dir = os.path.join(pyqt5_path, 'Qt', 'plugins')
            logger.debug(f"Intentando ruta alternativa: {alt_plugin_dir}")
            if os.path.exists(alt_plugin_dir):
                plugin_dir = alt_plugin_dir
                logger.info(f"✓ Plugin directory encontrado en ruta alternativa: {plugin_dir}")
            else:
                logger.error(f"No se encontró directorio de plugins en ninguna ruta")
                return False
            
    except ImportError as e:
        logger.error(f"PyQt5 not available: {e}")
        return False
    
    # Verificar plugins específicos requeridos
    logger.debug("Paso 2: Verificando plugins de plataforma...")
    platforms_dir = os.path.join(plugin_dir, 'platforms')
    logger.debug(f"Buscando platforms en: {platforms_dir}")
    
    if not os.path.exists(platforms_dir):
        logger.error(f"Platforms directory not found: {platforms_dir}")
        return False
    
    # Buscar cualquier plugin de plataforma disponible
    available_plugins = []
    if os.path.exists(platforms_dir):
        for file in os.listdir(platforms_dir):
            if file.endswith('.so'):
                available_plugins.append(file)
    
    logger.debug(f"Plugins encontrados: {available_plugins}")
    
    if not available_plugins:
        logger.error("No platform plugins found")
        return False
    
    logger.info(f"✓ Available platform plugins: {available_plugins}")
    
    # Verificar importaciones básicas de PyQt5
    logger.debug("Paso 3: Verificando importaciones básicas de PyQt5...")
    try:
        import PyQt5.QtWidgets
        logger.debug("✓ PyQt5.QtWidgets importado")
        import PyQt5.QtCore
        logger.debug("✓ PyQt5.QtCore importado")
        import PyQt5.QtGui
        logger.debug("✓ PyQt5.QtGui importado")
    except ImportError as e:
        logger.error(f"PyQt5 import error: {e}")
        return False
    
    # Configurar ruta de plugins
    logger.debug("Paso 4: Configurando variables de entorno...")
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_dir
    logger.debug(f"✓ QT_QPA_PLATFORM_PLUGIN_PATH = {plugin_dir}")
    
    # Verificar DISPLAY
    display = os.environ.get('DISPLAY', 'No configurado')
    logger.debug(f"DISPLAY = {display}")
    
    logger.info("✓ Qt funcional - verificaciones básicas completadas")
    return True


def create_application():
    """Crea y configura la aplicación PyQt5 solo si Qt está funcional"""
    
    logger = get_logger("application")
    
    # Verificar si Qt está realmente funcional
    if not is_qt_functional():
        logger.warning("Qt no está funcional en este entorno - se usará modo headless")
        return None
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QIcon
        import PyQt5
        
        # Configurar variables de entorno con la ruta correcta de PyQt5
        pyqt5_path = os.path.dirname(PyQt5.__file__)
        plugin_dir = os.path.join(pyqt5_path, 'Qt5', 'plugins')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_dir
        
        # Endurecer entorno de QtWebEngine para evitar crashes en entornos restringidos
        os.environ.setdefault('QTWEBENGINE_DISABLE_SANDBOX', '1')
        os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--no-sandbox --disable-gpu --disable-software-rasterizer')
        
        # Asegurar XDG_RUNTIME_DIR válido
        runtime_dir = os.environ.get('XDG_RUNTIME_DIR', '')
        if not runtime_dir or not os.path.exists(runtime_dir):
            tmp_runtime = os.path.join('/tmp', 'sigec-qt-runtime')
            try:
                os.makedirs(tmp_runtime, exist_ok=True)
            except Exception:
                tmp_runtime = '/tmp'
            os.environ['XDG_RUNTIME_DIR'] = tmp_runtime
        
        # FORZAR el uso del display real (no offscreen)
        if 'QT_QPA_PLATFORM' in os.environ:
            del os.environ['QT_QPA_PLATFORM']
        
        # Configurar display para GUI visible
        if 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':0'
        
        current_platform = os.environ.get('QT_QPA_PLATFORM', 'xcb')
        logger.info(f"Creando aplicación Qt con plataforma: {current_platform}")
        logger.info(f"Usando plugins de: {plugin_dir}")
        logger.info(f"Display configurado: {os.environ.get('DISPLAY', 'No configurado')}")
        
        # IMPORTANTE: Configurar atributos ANTES de crear QApplication
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        
        # Configurar atributo para QtWebEngine ANTES de crear QApplication
        if hasattr(Qt, 'AA_ShareOpenGLContexts'):
            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
            logger.info("✓ Qt::AA_ShareOpenGLContexts configurado")
        
        # Crear aplicación
        app = QApplication(sys.argv)
        
        # Configurar propiedades básicas
        app.setApplicationName("SIGeC-Balistica")
        app.setApplicationVersion("2.0.0")
        
        # Importar QtWebEngineWidgets DESPUÉS de crear QApplication
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            logger.info("✓ QtWebEngineWidgets importado correctamente")
            # Probar instanciación para confirmar funcionalidad real
            try:
                _tmp_webview = QWebEngineView()
                _tmp_webview.deleteLater()
                logger.debug("✓ QWebEngineView instanciado correctamente")
            except Exception as e:
                logger.warning(f"QWebEngineView falló al instanciarse, aplicando fallback: {e}")
                os.environ['SIGEC_DISABLE_WEBENGINE'] = '1'
        except ImportError:
            logger.warning("QtWebEngineWidgets no disponible, algunas funciones pueden estar limitadas")
            os.environ['SIGEC_DISABLE_WEBENGINE'] = '1'
        
        logger.info("✓ Aplicación PyQt5 creada exitosamente")
        return app
        
    except Exception as e:
        logger.error(f"Error creando aplicación PyQt5: {e}")
        logger.error(traceback.format_exc())
        return None

def create_main_window():
    """Crea la ventana principal"""
    
    logger = get_logger("main_window")
    logger.info("=== INICIANDO create_main_window() ===")
    
    try:
        logger.debug("Paso 1: Intentando importar QApplication...")
        from PyQt5.QtWidgets import QApplication
        logger.debug("✓ QApplication importado exitosamente.")
        
        logger.debug("Paso 2: Verificando que gui package existe...")
        import gui
        logger.debug(f"✓ gui package encontrado en: {gui.__file__}")
        
        logger.debug("Paso 3: Intentando importar gui.main_window module...")
        import gui.main_window
        logger.debug(f"✓ gui.main_window module importado desde: {gui.main_window.__file__}")
        
        logger.debug("Paso 4: Intentando importar MainWindow class...")
        from gui.main_window import MainWindow
        logger.debug("✓ MainWindow class importada exitosamente.")
        
        logger.debug("Paso 5: Intentando importar gui.styles module...")
        import gui.styles
        logger.debug(f"✓ gui.styles module importado desde: {gui.styles.__file__}")
        
        logger.debug("Paso 6: Intentando importar apply_SIGeC_Balistica_theme...")
        from gui.styles import apply_SIGeC_Balistica_theme
        logger.debug("✓ apply_SIGeC_Balistica_theme importado exitosamente.")
        
        logger.debug("✓ Todas las importaciones de GUI exitosas.")
        
        # Aplicar tema
        logger.debug("Paso 7: Obteniendo instancia de QApplication...")
        app = QApplication.instance()
        logger.debug(f"✓ QApplication instance: {app}")
        
        logger.debug("Paso 8: Aplicando tema SIGeC-Balistica...")
        apply_SIGeC_Balistica_theme(app)
        logger.debug("✓ Tema aplicado exitosamente.")
        
        # Crear ventana principal
        logger.debug("Paso 9: Creando instancia de MainWindow...")
        window = MainWindow()
        logger.debug(f"✓ Instancia de MainWindow creada: {window}")
        
        logger.info("✓ Ventana principal creada exitosamente")
        logger.debug(f"create_main_window() retornando: {window}")
        return window
        
    except ImportError as e:
        logger.error(f"ERROR DE IMPORTACIÓN en create_main_window(): {e}")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Argumentos del error: {e.args}")
        logger.error("Stack trace completo:")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"ERROR GENERAL en create_main_window(): {e}")
        logger.error(f"Tipo de error: {type(e).__name__}")
        logger.error(f"Argumentos del error: {e.args}")
        logger.error("Stack trace completo:")
        logger.error(traceback.format_exc())
        return None

def setup_exception_handling():
    """Configura el manejo global de excepciones"""
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Maneja excepciones no capturadas"""
        
        if issubclass(exc_type, KeyboardInterrupt):
            # Permitir Ctrl+C normal
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log del error
        error_msg = f"Excepción no manejada: {exc_type.__name__}: {exc_value}"
        
        # Obtener logger si está disponible
        try:
            current_logger = get_logger("exception_handler")
            current_logger.critical(error_msg)
            current_logger.critical("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        except:
            # Fallback a print si el logger no está disponible
            print(f"CRITICAL: {error_msg}")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        # Intentar mostrar diálogo de error si PyQt5 está disponible
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Error Crítico")
                msg.setText("Se ha producido un error inesperado")
                msg.setDetailedText(f"{exc_type.__name__}: {exc_value}")
                msg.exec_()
        except Exception:
            # Si no se puede mostrar el diálogo, al menos imprimir
            print(f"Error crítico: {exc_type.__name__}: {exc_value}")
    
    # Instalar el manejador
    sys.excepthook = handle_exception
    
    # Log de confirmación si hay logger disponible
    try:
        current_logger = get_logger("main")
        current_logger.info("✓ Manejo de excepciones configurado")
    except:
        pass  # No hacer nada si el logger no está disponible

def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    
    parser = argparse.ArgumentParser(
        description="SIGeC-Balistica - Sistema de Evaluación Automatizada de Características Balisticas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                    # Ejecutar con configuración por defecto
  python main.py --debug           # Ejecutar en modo debug
  python main.py --headless        # Ejecutar sin GUI (solo backend)
  python main.py --config custom.yaml  # Usar configuración personalizada
        """
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Ejecutar en modo debug (logging detallado)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Ejecutar sin GUI (solo backend)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Archivo de configuración personalizado'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Archivo de log personalizado'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Ejecutar pruebas de integración'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='SIGeC-Balistica 2.0.0'
    )
    
    return parser.parse_args()

def run_tests():
    """Ejecuta las pruebas de integración"""
    
    logger = get_logger("tests")
    
    logger.info("Ejecutando pruebas de integración...")
    
    try:
        import subprocess
        import sys
        
        # Ejecutar pytest directamente
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--maxfail=5"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            logger.info("✅ Todas las pruebas pasaron exitosamente")
            logger.info(f"Salida de pytest:\n{result.stdout}")
            return 0
        else:
            logger.error("❌ Algunas pruebas fallaron")
            logger.error(f"Error de pytest:\n{result.stderr}")
            logger.error(f"Salida de pytest:\n{result.stdout}")
            return 1
            
    except Exception as e:
        logger.error(f"Error ejecutando pruebas: {e}")
        return 1

def run_headless():
    """Ejecuta la aplicación en modo headless (sin GUI)"""
    
    logger = get_logger("headless")
    
    logger.info("Ejecutando en modo headless...")
    
    try:
        from gui.backend_integration import get_backend_integration
        
        # Inicializar backend
        backend = get_backend_integration()
        
        # Verificar estado del sistema
        status = backend.get_system_status()
        logger.info(f"Estado del sistema: {status}")
        
        # Mantener la aplicación corriendo
        logger.info("Backend inicializado. Presione Ctrl+C para salir.")
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Cerrando aplicación...")
            return 0
            
    except Exception as e:
        logger.error(f"Error en modo headless: {e}")
        return 1

def main():
    """Función principal"""
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar logging según argumentos
    from utils.logger import setup_logging
    log_level_str = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level_str, log_file=args.log_file)
    logger = get_logger("main")
    
    # DEBUG: Verificar argumentos inmediatamente después de configurar logging
    logger.info("=" * 60)
    logger.info("SIGeC-Balistica - Sistema de Evaluación Automatizada")
    logger.info("Versión 0.1.3")
    logger.info("=" * 60)
    logger.debug("=== DEBUG: ARGUMENTOS PARSEADOS ===")
    logger.debug(f"args.debug = {args.debug}")
    logger.debug(f"args.headless = {args.headless}")
    logger.debug(f"args.test = {args.test}")
    logger.debug(f"log_level_str = {log_level_str}")
    logger.debug(f"Todos los argumentos: {vars(args)}")
    logger.debug("=== FIN DEBUG ARGUMENTOS ===")
    logger.info("Continuando con inicialización del sistema...")
    
    # Configurar limpieza al salir
    def cleanup():
        """Función de limpieza al salir"""
        logger.info("Ejecutando limpieza del sistema...")
        stop_monitoring()
        logger.info("Limpieza completada")
    
    atexit.register(cleanup)
    
    try:
        # Configurar manejo de excepciones
        setup_exception_handling()
        logger.info("✓ Manejo de excepciones configurado")
        
        # Verificar requisitos del sistema
        logger.info("Verificando requisitos del sistema...")
        try:
            if not check_system_requirements():
                logger.error("Los requisitos del sistema no se cumplen")
                return 1
            logger.debug("✓ Requisitos del sistema verificados exitosamente")
        except Exception as e:
            logger.error(f"Error verificando requisitos del sistema: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Verificar dependencias
        logger.info("Verificando dependencias...")
        try:
            if not check_dependencies():
                logger.error("Dependencias faltantes")
                return 1
            logger.debug("✓ Dependencias verificadas exitosamente")
        except Exception as e:
            logger.error(f"Error verificando dependencias: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Configurar entorno
        logger.debug("Llamando a setup_environment()...")
        try:
            setup_environment()
            logger.debug("✓ setup_environment() completado exitosamente")
        except Exception as e:
            logger.error(f"Error en setup_environment(): {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-lanzar para que sea manejada por el except general
        
        logger.debug("✓ Entorno configurado exitosamente")
        
        # Inicializar sistema de monitoreo
        logger.debug("Iniciando configuración del sistema de monitoreo...")
        try:
            if MONITORING_AVAILABLE:
                logger.info("Inicializando sistema de monitoreo...")
                monitoring_config = {
                    'metrics_interval': 30,
                    'alert_check_interval': 60,
                    'dashboard_port': 5001,
                    'enable_dashboard': True
                }
                
                if initialize_monitoring(monitoring_config):
                    if start_monitoring():
                        logger.info("✓ Sistema de monitoreo iniciado")
                        logger.info("Dashboard disponible en: http://localhost:5001")
                    else:
                        logger.warning("No se pudo iniciar el sistema de monitoreo")
                else:
                    logger.warning("No se pudo inicializar el sistema de monitoreo")
            else:
                logger.info("Sistema de monitoreo no disponible")
        except Exception as e:
            logger.error(f"Error en inicialización del monitoreo: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("Continuando sin sistema de monitoreo...")
        
        logger.debug("✓ Configuración de monitoreo completada")
        logger.debug("Procediendo a verificar modo de ejecución...")
        
        # Ejecutar según modo
        logger.debug(f"=== VERIFICANDO MODO DE EJECUCIÓN ===")
        logger.debug(f"args.test = {args.test}")
        logger.debug(f"args.headless = {args.headless}")
        logger.debug(f"Argumentos completos: {vars(args)}")
        
        if args.test:
            logger.info("=== EJECUTANDO EN MODO TEST ===")
            return run_tests()
        
        elif args.headless:
            logger.info("=== EJECUTANDO EN MODO HEADLESS (SOLICITADO) ===")
            return run_headless()
        
        else:
            # Modo GUI normal
            logger.info("=== INICIANDO MODO GUI ===")
            logger.info("Iniciando interfaz gráfica...")
            
            # Verificar Qt antes de crear la aplicación
            logger.debug("=== VERIFICANDO QT FUNCIONAL ===")
            try:
                qt_functional = is_qt_functional()
                logger.debug(f"Resultado is_qt_functional(): {qt_functional}")
                if not qt_functional:
                    logger.warning("Qt no está funcionando correctamente")
                    logger.warning("=== EJECUTANDO EN MODO HEADLESS (FALLBACK QT) ===")
                    return run_headless()
            except Exception as e:
                logger.error(f"Error verificando Qt: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.warning("=== EJECUTANDO EN MODO HEADLESS (FALLBACK EXCEPCIÓN QT) ===")
                return run_headless()
            
            # Crear aplicación PyQt5
            logger.debug("=== CREANDO APLICACIÓN QT ===")
            app = create_application()
            logger.debug(f"create_application() returned: {app}")
            if app is None:
                logger.warning("No se pudo crear la aplicación PyQt5. Ejecutando en modo headless como fallback...")
                return run_headless()
            
            # Crear ventana principal
            logger.debug("=== CREANDO VENTANA PRINCIPAL ===")
            window = create_main_window()
            logger.debug(f"create_main_window() returned: {window}")
            if window is None:
                logger.warning("No se pudo crear la ventana principal. Ejecutando en modo headless como fallback...")
                return run_headless()
            
            try:
                # Mostrar ventana
                logger.debug("=== MOSTRANDO VENTANA PRINCIPAL ===")
                window.show()
                
                # Centrar ventana en pantalla
                screen = app.primaryScreen().geometry()
                window_size = window.geometry()
                x = (screen.width() - window_size.width()) // 2
                y = (screen.height() - window_size.height()) // 2
                window.move(x, y)
                
                logger.info("✓ Aplicación iniciada exitosamente")
                if MONITORING_AVAILABLE:
                    logger.info("Sistema de monitoreo activo - Dashboard: http://localhost:5001")
                logger.info("Presione Ctrl+C en la consola o cierre la ventana para salir")
                
                # Ejecutar bucle principal
                logger.debug("=== INICIANDO BUCLE PRINCIPAL QT ===")
                try:
                    exit_code = app.exec_()
                    logger.info(f"Aplicación cerrada con código: {exit_code}")
                    return exit_code
                
                except KeyboardInterrupt:
                    logger.info("Aplicación interrumpida por el usuario")
                    return 0
                    
            except Exception as e:
                logger.warning(f"Error mostrando la interfaz gráfica: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                logger.warning("=== EJECUTANDO EN MODO HEADLESS (FALLBACK GUI) ===")
                return run_headless()
    
    except Exception as e:
        logger.critical(f"=== ERROR CRÍTICO EN MAIN() ===")
        logger.critical(f"Error crítico en main(): {e}")
        logger.critical(traceback.format_exc())
        logger.critical("=== EJECUTANDO FALLBACK A HEADLESS MODE DEBIDO A EXCEPCIÓN ===")
        return run_headless()
    
    finally:
        logger.info("Finalizando SIGeC-Balistica...")
        # La limpieza se ejecutará automáticamente por atexit

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)