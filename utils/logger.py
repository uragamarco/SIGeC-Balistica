"""
Sistema de Logging
Sistema Balístico Forense MVP

Configuración centralizada de logging usando loguru
"""

import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5,
    console_output: bool = True
) -> None:
    """
    Configura el sistema de logging
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta del archivo de log (opcional)
        max_file_size: Tamaño máximo del archivo de log
        backup_count: Número de archivos de backup
        console_output: Si mostrar logs en consola
    """
    
    # Remover configuración por defecto de loguru
    logger.remove()
    
    # Configurar salida a consola si está habilitada
    if console_output:
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
    
    # Configurar archivo de log si se especifica
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=max_file_size,
            retention=backup_count,
            compression="zip",
            encoding="utf-8"
        )
    
    # Configurar logging estándar de Python para que use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Obtener el nivel correspondiente de loguru
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Encontrar el frame del caller
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Reemplazar handlers de logging estándar
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configurar loggers específicos
    for logger_name in ["opencv", "PyQt5", "faiss", "sklearn"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info(f"Sistema de logging configurado - Nivel: {log_level}")

def get_logger(name: str = None):
    """
    Obtiene un logger con el nombre especificado
    
    Args:
        name: Nombre del logger (opcional)
    
    Returns:
        Logger configurado
    """
    if name:
        return logger.bind(name=name)
    return logger

class LoggerMixin:
    """
    Mixin para agregar capacidades de logging a cualquier clase
    """
    
    @property
    def logger(self):
        """Retorna un logger con el nombre de la clase"""
        return get_logger(self.__class__.__name__)

# Funciones de conveniencia para logging
def log_debug(message: str, **kwargs):
    """Log de debug"""
    logger.debug(message, **kwargs)

def log_info(message: str, **kwargs):
    """Log de información"""
    logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log de advertencia"""
    logger.warning(message, **kwargs)

def log_error(message: str, **kwargs):
    """Log de error"""
    logger.error(message, **kwargs)

def log_critical(message: str, **kwargs):
    """Log crítico"""
    logger.critical(message, **kwargs)

def log_exception(message: str, **kwargs):
    """Log de excepción con traceback"""
    logger.exception(message, **kwargs)

# Decorador para logging automático de funciones
def log_function_call(func):
    """
    Decorador que registra las llamadas a funciones
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Llamando función: {func_name}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Función {func_name} completada exitosamente")
            return result
        except Exception as e:
            logger.error(f"Error en función {func_name}: {e}")
            raise
    
    return wrapper

# Contexto para logging de operaciones
class LogOperation:
    """
    Context manager para logging de operaciones
    """
    
    def __init__(self, operation_name: str, log_level: str = "INFO"):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        from time import time
        self.start_time = time()
        logger.log(self.log_level, f"Iniciando operación: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from time import time
        duration = time() - self.start_time
        
        if exc_type is None:
            logger.log(self.log_level, f"Operación completada: {self.operation_name} ({duration:.2f}s)")
        else:
            logger.error(f"Error en operación: {self.operation_name} ({duration:.2f}s) - {exc_val}")
        
        return False  # No suprimir excepciones