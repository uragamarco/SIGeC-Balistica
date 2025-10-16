#!/usr/bin/env python3
"""
Configuración de Producción - SIGeC-Balistica
Configuraciones optimizadas para entorno de producción
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import ssl
import secrets

# Importar configuración unificada de base de datos
from config.unified_config import DatabaseConfig


@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    secret_key: str = ""  # Se genera automáticamente
    jwt_secret_key: str = ""  # Se genera automáticamente
    jwt_expiration_hours: int = 24
    password_min_length: int = 12
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    csrf_protection: bool = True
    secure_cookies: bool = True
    https_only: bool = True
    content_security_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    
    def __post_init__(self):
        """Generar claves secretas si no están definidas"""
        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(32)
        if not self.jwt_secret_key:
            self.jwt_secret_key = secrets.token_urlsafe(32)


@dataclass
class PerformanceConfig:
    """Configuración de rendimiento"""
    # Cache
    cache_enabled: bool = True
    cache_max_size: int = 2000
    cache_ttl_seconds: int = 3600
    
    # Procesamiento paralelo
    max_workers: int = 8
    batch_size: int = 10
    use_multiprocessing: bool = True
    
    # Límites de recursos
    max_image_size_mb: int = 50
    max_batch_images: int = 100
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 300
    
    # Optimizaciones de algoritmos
    lbp_optimization: bool = True
    similarity_cache: bool = True
    database_query_cache: bool = True
    
    # Compresión
    enable_gzip: bool = True
    gzip_level: int = 6
    
    def __post_init__(self):
        """Ajustar configuración basada en recursos del sistema"""
        cpu_count = os.cpu_count() or 4
        self.max_workers = min(self.max_workers, cpu_count * 2)


@dataclass
class LoggingConfig:
    """Configuración de logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "/var/log/sigec-balistica/app.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = False  # Deshabilitado en producción
    enable_syslog: bool = True
    syslog_address: str = "/dev/log"
    
    # Logging específico
    log_sql_queries: bool = False
    log_performance_metrics: bool = True
    log_security_events: bool = True
    log_errors_to_file: bool = True


@dataclass
class MonitoringConfig:
    """Configuración de monitoreo"""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_port: int = 8080
    
    # Métricas
    collect_system_metrics: bool = True
    collect_application_metrics: bool = True
    metrics_interval_seconds: int = 60
    
    # Alertas
    enable_alerts: bool = True
    alert_email: str = "admin@sigec-balistica.com"
    smtp_server: str = "localhost"
    smtp_port: int = 587
    
    # Umbrales de alerta
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    response_time_threshold_ms: float = 5000.0
    error_rate_threshold_percent: float = 5.0


@dataclass
class BackupConfig:
    """Configuración de respaldos"""
    enabled: bool = True
    backup_directory: str = "/var/backups/sigec-balistica"
    
    # Respaldo de base de datos
    db_backup_enabled: bool = True
    db_backup_schedule: str = "0 2 * * *"  # Cron: 2 AM diario
    db_backup_retention_days: int = 30
    
    # Respaldo de archivos
    file_backup_enabled: bool = True
    file_backup_schedule: str = "0 3 * * 0"  # Cron: 3 AM semanal
    file_backup_retention_weeks: int = 12
    
    # Compresión
    compress_backups: bool = True
    compression_level: int = 6


@dataclass
class ProductionConfig:
    """Configuración principal de producción"""
    # Información del entorno
    environment: str = "production"
    debug: bool = False
    testing: bool = False
    
    # Configuraciones específicas
    database: DatabaseConfig = None
    security: SecurityConfig = None
    performance: PerformanceConfig = None
    logging: LoggingConfig = None
    monitoring: MonitoringConfig = None
    backup: BackupConfig = None
    
    # Configuraciones de aplicación
    app_name: str = "SIGeC-Balistica"
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    
    # Directorios
    base_directory: str = "/opt/sigec-balistica"
    data_directory: str = "/var/lib/sigec-balistica"
    temp_directory: str = "/tmp/sigec-balistica"
    log_directory: str = "/var/log/sigec-balistica"
    
    def __post_init__(self):
        """Inicializar configuraciones por defecto"""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.backup is None:
            self.backup = BackupConfig()
    
    def validate_config(self) -> List[str]:
        """Validar configuración y retornar errores"""
        errors = []
        
        # Validar base de datos
        if not self.database.password and not os.getenv('DB_PASSWORD'):
            errors.append("Database password not configured")
        
        # Validar directorios
        for directory in [self.data_directory, self.log_directory]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, mode=0o755, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {e}")
        
        # Validar configuración de seguridad
        if len(self.security.secret_key) < 32:
            errors.append("Secret key too short (minimum 32 characters)")
        
        # Validar configuración de rendimiento
        if self.performance.max_image_size_mb > 100:
            errors.append("Max image size too large (recommended: <= 100MB)")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return asdict(self)
    
    def save_to_file(self, file_path: str):
        """Guardar configuración a archivo"""
        config_dict = self.to_dict()
        
        # Remover información sensible
        if 'password' in config_dict.get('database', {}):
            config_dict['database']['password'] = "***HIDDEN***"
        if 'secret_key' in config_dict.get('security', {}):
            config_dict['security']['secret_key'] = "***HIDDEN***"
        if 'jwt_secret_key' in config_dict.get('security', {}):
            config_dict['security']['jwt_secret_key'] = "***HIDDEN***"
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ProductionConfig':
        """Cargar configuración desde archivo"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Crear instancias de configuraciones específicas
        database_config = DatabaseConfig(**config_dict.get('database', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        performance_config = PerformanceConfig(**config_dict.get('performance', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        backup_config = BackupConfig(**config_dict.get('backup', {}))
        
        # Crear configuración principal
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['database', 'security', 'performance', 
                                  'logging', 'monitoring', 'backup']}
        
        return cls(
            database=database_config,
            security=security_config,
            performance=performance_config,
            logging=logging_config,
            monitoring=monitoring_config,
            backup=backup_config,
            **main_config
        )


class ProductionConfigManager:
    """Gestor de configuración de producción"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "/etc/sigec-balistica/production.json"
        self._config = None
    
    def get_config(self) -> ProductionConfig:
        """Obtener configuración de producción"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> ProductionConfig:
        """Cargar configuración"""
        if os.path.exists(self.config_file):
            try:
                return ProductionConfig.load_from_file(self.config_file)
            except Exception as e:
                logging.warning(f"Error loading config file {self.config_file}: {e}")
        
        # Crear configuración por defecto
        config = ProductionConfig()
        
        # Cargar desde variables de entorno
        self._load_from_environment(config)
        
        return config
    
    def _load_from_environment(self, config: ProductionConfig):
        """Cargar configuración desde variables de entorno"""
        # Base de datos
        if os.getenv('DB_HOST'):
            config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            config.database.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            config.database.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            config.database.password = os.getenv('DB_PASSWORD')
        
        # Aplicación
        if os.getenv('APP_HOST'):
            config.app_host = os.getenv('APP_HOST')
        if os.getenv('APP_PORT'):
            config.app_port = int(os.getenv('APP_PORT'))
        
        # Seguridad
        if os.getenv('SECRET_KEY'):
            config.security.secret_key = os.getenv('SECRET_KEY')
        if os.getenv('JWT_SECRET_KEY'):
            config.security.jwt_secret_key = os.getenv('JWT_SECRET_KEY')
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            config.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            config.logging.file_path = os.getenv('LOG_FILE')
    
    def save_config(self, config: ProductionConfig):
        """Guardar configuración"""
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Guardar configuración
        config.save_to_file(self.config_file)
        self._config = config
    
    def validate_production_environment(self) -> List[str]:
        """Validar entorno de producción"""
        config = self.get_config()
        errors = config.validate_config()
        
        # Validaciones adicionales para producción
        
        # Verificar SSL/TLS
        if not config.security.https_only:
            errors.append("HTTPS should be enabled in production")
        
        # Verificar debug deshabilitado
        if config.debug:
            errors.append("Debug mode should be disabled in production")
        
        # Verificar logging
        if config.logging.enable_console:
            errors.append("Console logging should be disabled in production")
        
        # Verificar monitoreo
        if not config.monitoring.enabled:
            errors.append("Monitoring should be enabled in production")
        
        # Verificar respaldos
        if not config.backup.enabled:
            errors.append("Backups should be enabled in production")
        
        return errors
    
    def setup_logging(self):
        """Configurar logging para producción"""
        config = self.get_config()
        
        # Crear directorio de logs
        os.makedirs(os.path.dirname(config.logging.file_path), exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, config.logging.level.upper()),
            format=config.logging.format,
            handlers=[
                logging.FileHandler(config.logging.file_path),
                logging.StreamHandler() if config.logging.enable_console else logging.NullHandler()
            ]
        )
        
        # Configurar rotación de logs
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.logging.format))
        
        # Obtener logger raíz y agregar handler
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


# Instancia global del gestor de configuración
config_manager = ProductionConfigManager()


def get_production_config() -> ProductionConfig:
    """Obtener configuración de producción"""
    return config_manager.get_config()


def setup_production_environment():
    """Configurar entorno de producción"""
    config = get_production_config()
    
    # Configurar logging
    config_manager.setup_logging()
    
    # Validar configuración
    errors = config_manager.validate_production_environment()
    if errors:
        logging.error("Production configuration errors:")
        for error in errors:
            logging.error(f"  - {error}")
        raise ValueError("Invalid production configuration")
    
    # Crear directorios necesarios
    for directory in [config.data_directory, config.temp_directory, config.log_directory]:
        os.makedirs(directory, mode=0o755, exist_ok=True)
    
    logging.info("Production environment configured successfully")
    return config


if __name__ == "__main__":
    # Ejemplo de uso
    print("=== CONFIGURACIÓN DE PRODUCCIÓN ===")
    
    # Crear configuración
    config = ProductionConfig()
    
    # Validar configuración
    errors = config.validate_config()
    if errors:
        print("Errores de configuración:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Configuración válida")
    
    # Mostrar configuración (sin información sensible)
    print("\nConfiguración actual:")
    print(f"  Entorno: {config.environment}")
    print(f"  Host: {config.app_host}:{config.app_port}")
    print(f"  Base de datos: {config.database.host}:{config.database.port}")
    print(f"  Cache habilitado: {config.performance.cache_enabled}")
    print(f"  Monitoreo habilitado: {config.monitoring.enabled}")
    print(f"  Respaldos habilitados: {config.backup.enabled}")
    
    # Guardar configuración de ejemplo
    example_config_file = "/tmp/sigec_production_example.json"
    config.save_to_file(example_config_file)
    print(f"\nConfiguración guardada en: {example_config_file}")