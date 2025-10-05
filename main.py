#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIGeC-Balisticar - Sistema Experto de Análisis de Cartuchos y Balas Automático
=====================================================================

Aplicación principal que integra todas las funcionalidades del sistema forense balístico:
- Análisis individual de cartuchos, balas y proyectiles
- Comparación directa y búsqueda en base de datos balística
- Gestión de base de datos de evidencia balística
- Generación de reportes forenses con estándares NIST y conclusiones AFTE

Autor: SIGeC-Balisticar Team
Fecha: Octubre 2025
Versión: 2.0.0
"""

import sys
import os
import logging
import argparse
import traceback
from pathlib import Path

# Añadir el directorio del proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar sistema de logging centralizado
from utils.logger import get_logger

# Logger inicial (se configurará en main())
logger = None

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
    os.environ['SIGeC-BalisticaR_ROOT'] = str(project_root)
    os.environ['SIGeC-BalisticaR_DATA'] = str(project_root / "data")
    
    logger.info("✓ Entorno configurado")

def create_application():
    """Crea y configura la aplicación PyQt5"""
    
    logger = get_logger("application")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt, QDir
        from PyQt5.QtGui import QIcon, QPixmap
        
        # Configurar atributos de Qt ANTES de crear la aplicación
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        if hasattr(Qt, 'AA_ShareOpenGLContexts'):
            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        
        # Crear aplicación
        app = QApplication(sys.argv)
        
        # Configurar propiedades de la aplicación
        app.setApplicationName("SIGeC-Balisticar")
        app.setApplicationDisplayName("SIGeC-Balisticar - Sistema de Evaluación Automatizada")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("SIGeC-Balisticar Team")
        app.setOrganizationDomain("SIGeC-Balisticar.org")
        
        # Configurar icono de la aplicación
        icon_path = project_root / "resources" / "icons" / "SIGeC-Balisticar.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
        
        logger.info("✓ Aplicación PyQt5 creada")
        return app
        
    except ImportError as e:
        logger.error(f"Error importando PyQt5: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creando aplicación: {e}")
        return None

def create_main_window():
    """Crea la ventana principal"""
    
    logger = get_logger("main_window")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from gui.styles import apply_SIGeC_Balistica_theme
        
        # Aplicar tema
        app = QApplication.instance()
        apply_SIGeC_Balistica_theme(app)
        
        # Crear ventana principal
        window = MainWindow()
        
        logger.info("✓ Ventana principal creada")
        return window
        
    except ImportError as e:
        logger.error(f"Error importando componentes GUI: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creando ventana principal: {e}")
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
        description="SIGeC-Balisticar - Sistema de Evaluación Automatizada de Características Balisticas",
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
        version='SIGeC-Balisticar 2.0.0'
    )
    
    return parser.parse_args()

def run_tests():
    """Ejecuta las pruebas de integración"""
    
    logger = get_logger("tests")
    
    logger.info("Ejecutando pruebas de integración...")
    
    try:
        from tests.gui.test_gui_integration import run_integration_tests
        
        success = run_integration_tests()
        
        if success:
            logger.info("✅ Todas las pruebas pasaron exitosamente")
            return 0
        else:
            logger.error("❌ Algunas pruebas fallaron")
            return 1
            
    except ImportError as e:
        logger.error(f"No se pudieron importar las pruebas: {e}")
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
    
    logger.info("=" * 60)
    logger.info("SIGeC-Balisticar - Sistema de Evaluación Automatizada")
    logger.info("Versión 0.1.3")
    logger.info("=" * 60)
    
    try:
        # Configurar manejo de excepciones
        setup_exception_handling()
        
        # Verificar requisitos del sistema
        if not check_system_requirements():
            logger.error("Los requisitos del sistema no se cumplen")
            return 1
        
        # Verificar dependencias
        if not check_dependencies():
            logger.error("Dependencias faltantes")
            return 1
        
        # Configurar entorno
        setup_environment()
        
        # Ejecutar según modo
        if args.test:
            return run_tests()
        
        elif args.headless:
            return run_headless()
        
        else:
            # Modo GUI normal
            logger.info("Iniciando interfaz gráfica...")
            
            # Crear aplicación PyQt5
            app = create_application()
            if app is None:
                logger.error("No se pudo crear la aplicación PyQt5")
                return 1
            
            # Crear ventana principal
            window = create_main_window()
            if window is None:
                logger.error("No se pudo crear la ventana principal")
                return 1
            
            # Mostrar ventana
            window.show()
            
            # Centrar ventana en pantalla
            screen = app.primaryScreen().geometry()
            window_size = window.geometry()
            x = (screen.width() - window_size.width()) // 2
            y = (screen.height() - window_size.height()) // 2
            window.move(x, y)
            
            logger.info("✓ Aplicación iniciada exitosamente")
            logger.info("Presione Ctrl+C en la consola o cierre la ventana para salir")
            
            # Ejecutar bucle principal
            try:
                exit_code = app.exec_()
                logger.info(f"Aplicación cerrada con código: {exit_code}")
                return exit_code
                
            except KeyboardInterrupt:
                logger.info("Aplicación interrumpida por el usuario")
                return 0
    
    except Exception as e:
        logger.critical(f"Error crítico en main(): {e}")
        logger.critical(traceback.format_exc())
        return 1
    
    finally:
        logger.info("Finalizando SIGeC-Balisticar...")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)