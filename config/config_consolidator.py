"""
Consolidador de Sistema de Configuración para SIGeC-Balistica
Sistema unificado para migrar y consolidar todas las configuraciones dispersas

Este módulo implementa:
- Detección automática de archivos de configuración legacy
- Migración automática a sistema unificado
- Validación y normalización de configuraciones
- Backup automático de configuraciones existentes
- Resolución de conflictos entre configuraciones

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import os
import json
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class ConfigSource:
    """Información sobre una fuente de configuración"""
    path: Path
    type: str  # 'yaml', 'json', 'py', 'env'
    priority: int  # Mayor número = mayor prioridad
    last_modified: datetime
    size_bytes: int
    is_legacy: bool = True
    migrated: bool = False
    backup_path: Optional[Path] = None
    
    def __hash__(self):
        """Hace la clase hashable para usar en sets."""
        return hash((str(self.path), self.type, self.priority))
    
    def __eq__(self, other):
        """Define igualdad para comparaciones."""
        if not isinstance(other, ConfigSource):
            return False
        return (str(self.path), self.type, self.priority) == (str(other.path), other.type, other.priority)

@dataclass
class ConfigConflict:
    """Conflicto entre configuraciones"""
    key: str
    sources: List[ConfigSource]
    values: List[Any]
    resolution: Optional[str] = None  # 'highest_priority', 'newest', 'manual'
    resolved_value: Optional[Any] = None

@dataclass
class MigrationReport:
    """Reporte de migración de configuraciones"""
    total_sources: int = 0
    migrated_sources: int = 0
    conflicts_found: int = 0
    conflicts_resolved: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    backup_paths: List[Path] = field(default_factory=list)
    migration_time: Optional[datetime] = None

class ConfigConsolidator:
    """
    Consolidador de configuraciones del sistema SIGeC-Balistica
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Inicializar consolidador de configuraciones
        
        Args:
            project_root: Directorio raíz del proyecto
        """
        self.project_root = project_root or Path.cwd()
        self.config_dir = self.project_root / "config"
        self.backup_dir = self.config_dir / "backups" / "migration"
        self.unified_config_path = self.config_dir / "unified_config.yaml"
        
        # Crear directorios necesarios
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Patrones de archivos de configuración
        self.config_patterns = {
            'yaml': ['*.yaml', '*.yml'],
            'json': ['*.json'],
            'py': ['*config*.py', '*settings*.py'],
            'env': ['.env*', '*.env']
        }
        
        # Prioridades por tipo y ubicación
        self.priority_rules = {
            'unified_config.yaml': 100,  # Máxima prioridad
            'config.yaml': 90,
            'settings.yaml': 80,
            'config.json': 70,
            'settings.json': 60,
            'config.py': 50,
            '.env': 40
        }
        
        # Mapeo de claves legacy a nuevas
        self.key_mappings = {
            # Database mappings
            'db_path': 'database.sqlite_path',
            'database_path': 'database.sqlite_path',
            'backup_enabled': 'database.backup_enabled',
            'backup_interval': 'database.backup_interval_hours',
            'faiss_path': 'database.faiss_index_path',
            
            # Image processing mappings
            'max_image_size': 'image_processing.max_image_size',
            'orb_features': 'image_processing.orb_features',
            'sift_features': 'image_processing.sift_features',
            'supported_formats': 'image_processing.supported_formats',
            'roi_method': 'image_processing.roi_detection_method',
            
            # GUI mappings
            'theme': 'gui.theme',
            'language': 'gui.language',
            'window_width': 'gui.window_width',
            'window_height': 'gui.window_height',
            'auto_save': 'gui.auto_save',
            
            # Logging mappings
            'log_level': 'logging.level',
            'log_path': 'logging.log_path',
            'log_file': 'logging.file_path',
            'max_log_size': 'logging.max_file_size',
            'log_backup_count': 'logging.backup_count',
            
            # Matching mappings
            'distance_threshold': 'matching.distance_threshold',
            'similarity_threshold': 'matching.similarity_threshold',
            'matcher_type': 'matching.matcher_type',
            'min_matches': 'matching.min_matches',
            'max_results': 'matching.max_results',
            
            # Deep learning mappings
            'dl_enabled': 'deep_learning.enabled',
            'dl_device': 'deep_learning.device',
            'batch_size': 'deep_learning.batch_size',
            'learning_rate': 'deep_learning.learning_rate',
            'epochs': 'deep_learning.epochs'
        }
        
        logger.info(f"ConfigConsolidator inicializado en: {self.project_root}")
    
    def discover_config_sources(self) -> List[ConfigSource]:
        """
        Descubrir todas las fuentes de configuración en el proyecto
        
        Returns:
            Lista de fuentes de configuración encontradas
        """
        sources = []
        
        # Buscar archivos de configuración
        for config_type, patterns in self.config_patterns.items():
            for pattern in patterns:
                for config_file in self.project_root.rglob(pattern):
                    # Excluir directorios de backup y temporales
                    if any(part in str(config_file) for part in ['backup', 'temp', '__pycache__', '.git']):
                        continue
                    
                    # Obtener información del archivo
                    try:
                        stat = config_file.stat()
                        priority = self._get_file_priority(config_file)
                        
                        source = ConfigSource(
                            path=config_file,
                            type=config_type,
                            priority=priority,
                            last_modified=datetime.fromtimestamp(stat.st_mtime),
                            size_bytes=stat.st_size,
                            is_legacy=config_file.name != 'unified_config.yaml'
                        )
                        
                        sources.append(source)
                        
                    except Exception as e:
                        logger.warning(f"Error procesando archivo {config_file}: {e}")
        
        # Ordenar por prioridad (mayor a menor)
        sources.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Encontradas {len(sources)} fuentes de configuración")
        return sources
    
    def _get_file_priority(self, config_file: Path) -> int:
        """Obtener prioridad de un archivo de configuración"""
        filename = config_file.name.lower()
        
        # Prioridad específica por nombre
        for pattern, priority in self.priority_rules.items():
            if pattern in filename:
                return priority
        
        # Prioridad por ubicación
        if 'config' in str(config_file.parent).lower():
            return 30
        elif config_file.parent == self.project_root:
            return 20
        else:
            return 10
    
    def load_config_source(self, source: ConfigSource) -> Dict[str, Any]:
        """
        Cargar configuración desde una fuente
        
        Args:
            source: Fuente de configuración
            
        Returns:
            Diccionario con la configuración cargada
        """
        try:
            if source.type == 'yaml':
                with open(source.path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            
            elif source.type == 'json':
                with open(source.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif source.type == 'py':
                return self._load_python_config(source.path)
            
            elif source.type == 'env':
                return self._load_env_config(source.path)
            
            else:
                logger.warning(f"Tipo de configuración no soportado: {source.type}")
                return {}
                
        except Exception as e:
            logger.error(f"Error cargando configuración desde {source.path}: {e}")
            return {}
    
    def _load_python_config(self, config_path: Path) -> Dict[str, Any]:
        """Cargar configuración desde archivo Python"""
        config = {}
        
        try:
            # Leer el archivo y extraer variables de configuración
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Buscar patrones de configuración comunes
            patterns = [
                r'(\w+)\s*=\s*(["\'].*?["\'])',  # Strings
                r'(\w+)\s*=\s*(\d+\.?\d*)',      # Numbers
                r'(\w+)\s*=\s*(True|False)',     # Booleans
                r'(\w+)\s*=\s*(\[.*?\])',        # Lists
                r'(\w+)\s*=\s*(\{.*?\})',        # Dicts
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                for key, value in matches:
                    if key.isupper() or 'config' in key.lower() or 'setting' in key.lower():
                        try:
                            # Evaluar el valor de forma segura
                            config[key.lower()] = eval(value)
                        except:
                            config[key.lower()] = value.strip('"\'')
            
        except Exception as e:
            logger.warning(f"Error procesando archivo Python {config_path}: {e}")
        
        return config
    
    def _load_env_config(self, env_path: Path) -> Dict[str, Any]:
        """Cargar configuración desde archivo .env"""
        config = {}
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip().strip('"\'')
                        
                        # Convertir valores comunes
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').isdigit():
                            value = float(value)
                        
                        config[key] = value
        
        except Exception as e:
            logger.warning(f"Error procesando archivo .env {env_path}: {e}")
        
        return config
    
    def normalize_config_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizar claves de configuración usando mapeos
        
        Args:
            config: Configuración original
            
        Returns:
            Configuración con claves normalizadas
        """
        normalized = {}
        
        def set_nested_value(target: Dict, key_path: str, value: Any):
            """Establecer valor en diccionario anidado"""
            keys = key_path.split('.')
            current = target
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
        
        # Procesar cada clave
        for key, value in config.items():
            # Buscar mapeo directo
            if key in self.key_mappings:
                new_key = self.key_mappings[key]
                set_nested_value(normalized, new_key, value)
            else:
                # Mantener clave original si no hay mapeo
                normalized[key] = value
        
        return normalized
    
    def detect_conflicts(self, sources: List[ConfigSource]) -> List[ConfigConflict]:
        """
        Detectar conflictos entre configuraciones
        
        Args:
            sources: Lista de fuentes de configuración
            
        Returns:
            Lista de conflictos detectados
        """
        conflicts = []
        key_sources = defaultdict(list)
        
        # Agrupar fuentes por clave de configuración
        for source in sources:
            config = self.load_config_source(source)
            normalized = self.normalize_config_keys(config)
            
            def collect_keys(data: Dict, prefix: str = ""):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, dict):
                        collect_keys(value, full_key)
                    else:
                        key_sources[full_key].append((source, value))
            
            collect_keys(normalized)
        
        # Detectar conflictos
        for key, source_values in key_sources.items():
            if len(source_values) > 1:
                # Verificar si hay valores diferentes
                values = [sv[1] for sv in source_values]
                if len(set(str(v) for v in values)) > 1:  # Convertir a string para comparar
                    conflict = ConfigConflict(
                        key=key,
                        sources=[sv[0] for sv in source_values],
                        values=values
                    )
                    conflicts.append(conflict)
        
        logger.info(f"Detectados {len(conflicts)} conflictos de configuración")
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[ConfigConflict]) -> List[ConfigConflict]:
        """
        Resolver conflictos automáticamente
        
        Args:
            conflicts: Lista de conflictos
            
        Returns:
            Lista de conflictos resueltos
        """
        resolved = []
        
        for conflict in conflicts:
            # Estrategia de resolución: mayor prioridad
            highest_priority_source = max(conflict.sources, key=lambda s: s.priority)
            highest_priority_index = conflict.sources.index(highest_priority_source)
            
            conflict.resolution = 'highest_priority'
            conflict.resolved_value = conflict.values[highest_priority_index]
            
            resolved.append(conflict)
            
            logger.debug(f"Conflicto resuelto para '{conflict.key}': "
                        f"valor '{conflict.resolved_value}' de {highest_priority_source.path}")
        
        return resolved
    
    def _resolve_conflicts(self, conflicts: List[ConfigConflict]) -> Dict[str, Any]:
        """Resuelve conflictos de configuración usando estrategias inteligentes."""
        resolved_config = {}
        
        for conflict in conflicts:
            key = conflict.key
            values = conflict.values
            sources = conflict.sources
            
            # Estrategia 1: Prioridad por tipo de fuente
            priority_order = ['env', 'yaml', 'json', 'py']
            best_source = None
            best_value = None
            
            for priority_type in priority_order:
                for i, source in enumerate(sources):
                    if source.type == priority_type:
                        best_source = source
                        best_value = values[i]
                        break
                if best_source:
                    break
            
            # Estrategia 2: Si no hay preferencia por tipo, usar prioridad numérica
            if not best_source:
                max_priority = max(source.priority for source in sources)
                for i, source in enumerate(sources):
                    if source.priority == max_priority:
                        best_source = source
                        best_value = values[i]
                        break
            
            # Estrategia 3: Fusión inteligente para diccionarios
            if isinstance(best_value, dict):
                merged_dict = {}
                for i, value in enumerate(values):
                    if isinstance(value, dict):
                        merged_dict.update(value)
                best_value = merged_dict
            
            # Asignar valor resuelto usando notación de puntos
            self._set_nested_value(resolved_config, key, best_value)
            
        return resolved_config
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Establece un valor en un diccionario anidado usando notación de puntos."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def create_backup(self, sources: List[ConfigSource]) -> List[Path]:
        """
        Crear backup de configuraciones existentes
        
        Args:
            sources: Fuentes a respaldar
            
        Returns:
            Lista de rutas de backup creadas
        """
        backup_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for source in sources:
            if source.is_legacy:
                # Crear estructura de directorios en backup
                relative_path = source.path.relative_to(self.project_root)
                backup_path = self.backup_dir / timestamp / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.copy2(source.path, backup_path)
                    source.backup_path = backup_path
                    backup_paths.append(backup_path)
                    
                    logger.debug(f"Backup creado: {source.path} -> {backup_path}")
                    
                except Exception as e:
                    logger.error(f"Error creando backup de {source.path}: {e}")
        
        logger.info(f"Creados {len(backup_paths)} backups en {self.backup_dir / timestamp}")
        return backup_paths
    
    def generate_unified_config(self, sources: List[ConfigSource], 
                              conflicts: List[ConfigConflict]) -> Dict[str, Any]:
        """
        Generar configuración unificada
        
        Args:
            sources: Fuentes de configuración
            conflicts: Conflictos resueltos
            
        Returns:
            Configuración unificada
        """
        unified = {}
        
        # Crear mapeo de resoluciones de conflictos
        conflict_resolutions = {c.key: c.resolved_value for c in conflicts}
        
        # Procesar todas las fuentes
        all_keys = set()
        source_configs = {}
        
        for source in sources:
            config = self.load_config_source(source)
            normalized = self.normalize_config_keys(config)
            source_configs[source] = normalized
            
            def collect_all_keys(data: Dict, prefix: str = ""):
                for key, value in data.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    all_keys.add(full_key)
                    
                    if isinstance(value, dict):
                        collect_all_keys(value, full_key)
            
            collect_all_keys(normalized)
        
        # Construir configuración unificada
        def set_nested_value(target: Dict, key_path: str, value: Any):
            keys = key_path.split('.')
            current = target
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
        
        for key in all_keys:
            if key in conflict_resolutions:
                # Usar valor resuelto del conflicto
                set_nested_value(unified, key, conflict_resolutions[key])
            else:
                # Buscar valor en fuente de mayor prioridad
                for source in sorted(sources, key=lambda s: s.priority, reverse=True):
                    config = source_configs.get(source, {})
                    
                    def get_nested_value(data: Dict, key_path: str) -> Tuple[bool, Any]:
                        keys = key_path.split('.')
                        current = data
                        
                        try:
                            for k in keys:
                                current = current[k]
                            return True, current
                        except (KeyError, TypeError):
                            return False, None
                    
                    found, value = get_nested_value(config, key)
                    if found:
                        set_nested_value(unified, key, value)
                        break
        
        # Agregar metadatos
        unified['_metadata'] = {
            'version': '1.0.0',
            'consolidated_at': datetime.now().isoformat(),
            'sources_count': len(sources),
            'conflicts_resolved': len(conflicts),
            'project_root': str(self.project_root)
        }
        
        return unified
    
    def save_unified_config(self, config: Dict[str, Any]) -> Path:
        """
        Guardar configuración unificada
        
        Args:
            config: Configuración unificada
            
        Returns:
            Ruta del archivo guardado
        """
        try:
            with open(self.unified_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2, sort_keys=True)
            
            logger.info(f"Configuración unificada guardada en: {self.unified_config_path}")
            return self.unified_config_path
            
        except Exception as e:
            logger.error(f"Error guardando configuración unificada: {e}")
            raise
    
    def consolidate_configurations(self, dry_run: bool = False) -> MigrationReport:
        """
        Consolidar todas las configuraciones del sistema
        
        Args:
            dry_run: Si True, solo simula la consolidación
            
        Returns:
            Reporte de la migración
        """
        report = MigrationReport(migration_time=datetime.now())
        
        try:
            logger.info("Iniciando consolidación de configuraciones...")
            
            # 1. Descubrir fuentes
            sources = self.discover_config_sources()
            report.total_sources = len(sources)
            
            if not sources:
                report.warnings.append("No se encontraron fuentes de configuración")
                return report
            
            # 2. Detectar conflictos
            conflicts = self.detect_conflicts(sources)
            report.conflicts_found = len(conflicts)
            
            # 3. Resolver conflictos
            resolved_conflicts = self.resolve_conflicts(conflicts)
            report.conflicts_resolved = len(resolved_conflicts)
            
            if not dry_run:
                # 4. Crear backups
                backup_paths = self.create_backup(sources)
                report.backup_paths = backup_paths
                
                # 5. Generar configuración unificada
                unified_config = self.generate_unified_config(sources, resolved_conflicts)
                
                # 6. Guardar configuración unificada
                self.save_unified_config(unified_config)
                
                # 7. Marcar fuentes como migradas
                for source in sources:
                    if source.is_legacy:
                        source.migrated = True
                        report.migrated_sources += 1
            
            logger.info(f"Consolidación completada. Migradas {report.migrated_sources} fuentes")
            
        except Exception as e:
            error_msg = f"Error durante consolidación: {e}"
            logger.error(error_msg)
            report.errors.append(error_msg)
        
        return report
    
    def validate_unified_config(self) -> List[str]:
        """
        Validar configuración unificada
        
        Returns:
            Lista de errores de validación
        """
        errors = []
        
        if not self.unified_config_path.exists():
            errors.append("Archivo de configuración unificada no existe")
            return errors
        
        try:
            with open(self.unified_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validaciones básicas
            required_sections = ['database', 'image_processing', 'matching', 'gui', 'logging']
            
            for section in required_sections:
                if section not in config:
                    errors.append(f"Sección requerida faltante: {section}")
            
            # Validaciones específicas
            if 'database' in config:
                db_config = config['database']
                if 'sqlite_path' not in db_config:
                    errors.append("database.sqlite_path es requerido")
            
            if 'logging' in config:
                log_config = config['logging']
                if 'level' in log_config:
                    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    if log_config['level'] not in valid_levels:
                        errors.append(f"logging.level debe ser uno de: {valid_levels}")
            
        except Exception as e:
            errors.append(f"Error validando configuración: {e}")
        
        return errors
    
    def cleanup_legacy_configs(self, confirm: bool = False) -> List[Path]:
        """
        Limpiar configuraciones legacy después de migración exitosa
        
        Args:
            confirm: Confirmación para eliminar archivos
            
        Returns:
            Lista de archivos eliminados
        """
        if not confirm:
            logger.warning("cleanup_legacy_configs requiere confirmación explícita")
            return []
        
        removed_files = []
        sources = self.discover_config_sources()
        
        for source in sources:
            if source.is_legacy and source.migrated and source.backup_path:
                try:
                    source.path.unlink()
                    removed_files.append(source.path)
                    logger.info(f"Archivo legacy eliminado: {source.path}")
                except Exception as e:
                    logger.error(f"Error eliminando {source.path}: {e}")
        
        return removed_files

# Funciones de conveniencia
def consolidate_system_config(project_root: Optional[Path] = None, 
                            dry_run: bool = False) -> MigrationReport:
    """
    Consolidar configuraciones del sistema
    
    Args:
        project_root: Directorio raíz del proyecto
        dry_run: Solo simular la consolidación
        
    Returns:
        Reporte de migración
    """
    consolidator = ConfigConsolidator(project_root)
    return consolidator.consolidate_configurations(dry_run)

def validate_system_config(project_root: Optional[Path] = None) -> List[str]:
    """
    Validar configuración unificada del sistema
    
    Args:
        project_root: Directorio raíz del proyecto
        
    Returns:
        Lista de errores de validación
    """
    consolidator = ConfigConsolidator(project_root)
    return consolidator.validate_unified_config()