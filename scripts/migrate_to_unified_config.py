#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Migración a Configuración Unificada - SIGeC-Balistica
========================================================

Este script migra automáticamente todas las configuraciones dispersas
del proyecto SIGeC-Balisticaal nuevo sistema de configuración unificado,
actualizando referencias hardcodeadas y consolidando archivos de configuración.

Funcionalidades:
- Migra config/unified_config.yaml y utils/config.py al sistema unificado
- Actualiza referencias hardcodeadas en todo el código
- Migra configuraciones de deep learning
- Actualiza imports y referencias de configuración
- Crea respaldos de archivos modificados
- Genera reporte de migración

Autor: SIGeC-BalisticaTeam
Fecha: Diciembre 2024
"""

import os
import sys
import re
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigMigrator:
    """Migrador de configuraciones al sistema unificado"""
    
    def __init__(self, project_root: str = "get_project_root()"):
        """
        Inicializa el migrador
        
        Args:
            project_root: Ruta raíz del proyecto
        """
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "migration_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos a migrar
        self.legacy_config_files = [
            self.project_root / "config/unified_config/unified_config.yaml",
            self.project_root / "tests" / "config/unified_config/unified_config.yaml",
            self.project_root / "utils" / "config.py"
        ]
        
        # Patrones de código a actualizar
        self.code_patterns = {
            # Imports de configuración legacy
            r'from utils\.config import.*': 'from config.unified_config import get_unified_config',
            r'import utils\.config.*': 'from config.unified_config import get_unified_config',
            r'from config.unified_config import get_unified_config.*': 'from config.unified_config import get_unified_config',
            
            # Referencias a get_unified_config() legacy
            r'Config\(\)': 'get_unified_config()',
            r'config\.Config\(\)': 'get_unified_config()',
            r'utils\.config\.Config\(\)': 'get_unified_config()',
            
            # Referencias a archivos de configuración hardcodeados
            r'"config\.yaml"': '"config/unified_config/unified_config.yaml"',
            r"'config\.yaml'": "'config/unified_config/unified_config.yaml'",
            r'config\.yaml': 'config/unified_config/unified_config.yaml',
            
            # Rutas hardcodeadas específicas
            r'config\.yaml': 'config/unified_config/unified_config.yaml',
            r'': '',  # Convertir a rutas relativas
            
            # Referencias a configuraciones específicas
            r'config\.database': 'config.database',
            r'config\.image_processing': 'config.image_processing',
            r'config\.matching': 'config.matching',
            r'config\.gui': 'config.gui',
            r'config\.logging': 'config.logging',
            r'config\.deep_learning': 'config.deep_learning',
        }
        
        # Archivos a excluir de la migración
        self.exclude_patterns = [
            r'\.git/',
            r'__pycache__/',
            r'\.pytest_cache/',
            r'\.venv/',
            r'venv/',
            r'node_modules/',
            r'migration_backups/',
            r'\.pyc$',
            r'\.pyo$',
            r'\.log$',
            r'\.tmp$'
        ]
        
        self.migration_report = {
            'start_time': datetime.now().isoformat(),
            'files_processed': [],
            'files_modified': [],
            'errors': [],
            'warnings': [],
            'summary': {}
        }
    
    def should_exclude_file(self, file_path: Path) -> bool:
        """Verifica si un archivo debe ser excluido de la migración"""
        file_str = str(file_path.relative_to(self.project_root))
        
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_str):
                return True
        
        return False
    
    def create_backup(self, file_path: Path) -> Path:
        """Crea respaldo de un archivo"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            backup_path = self.backup_dir / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"Respaldo creado: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creando respaldo de {file_path}: {e}")
            raise
    
    def migrate_legacy_config_files(self) -> None:
        """Migra archivos de configuración legacy"""
        logger.info("Migrando archivos de configuración legacy...")
        
        # Migrar config/unified_config.yaml principal
        main_config = self.project_root / "config/unified_config/unified_config.yaml"
        if main_config.exists():
            self.create_backup(main_config)
            
            try:
                with open(main_config, 'r', encoding='utf-8') as f:
                    legacy_data = yaml.safe_load(f)
                
                # El sistema unificado se encargará de la migración automática
                logger.info(f"Configuración legacy encontrada en {main_config}")
                self.migration_report['files_processed'].append(str(main_config))
                
            except Exception as e:
                error_msg = f"Error procesando {main_config}: {e}"
                logger.error(error_msg)
                self.migration_report['errors'].append(error_msg)
        
        # Migrar config/unified_config.yaml de tests
        test_config = self.project_root / "tests" / "config/unified_config/unified_config.yaml"
        if test_config.exists():
            self.create_backup(test_config)
            logger.info(f"Configuración de tests encontrada en {test_config}")
            self.migration_report['files_processed'].append(str(test_config))
    
    def update_python_imports(self, file_path: Path) -> bool:
        """Actualiza imports de Python en un archivo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            modified = False
            
            # Actualizar imports específicos
            import_updates = {
                'from config.unified_config import get_unified_config': 'from config.unified_config import get_unified_config',
                'from config.unified_config import get_unified_config': 'from config.unified_config import get_unified_config',
                'from config.unified_config import get_unified_config': 'from config.unified_config import get_unified_config',
                'from config.unified_config import get_unified_config': 'from config.unified_config import get_unified_config',
            }
            
            for old_import, new_import in import_updates.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True
                    logger.info(f"Actualizado import en {file_path}: {old_import} -> {new_import}")
            
            # Actualizar instanciación de Config
            config_instantiation_updates = {
                'get_unified_config()': 'get_unified_config()',
                'config.get_unified_config()': 'get_unified_config()',
                'utils.config.get_unified_config()': 'get_unified_config()',
            }
            
            for old_inst, new_inst in config_instantiation_updates.items():
                if old_inst in content:
                    content = content.replace(old_inst, new_inst)
                    modified = True
                    logger.info(f"Actualizada instanciación en {file_path}: {old_inst} -> {new_inst}")
            
            # Actualizar referencias a archivos de configuración
            config_file_updates = {
                '"config/unified_config/unified_config.yaml"': '"config/unified_config/unified_config.yaml"',
                "'config/unified_config/unified_config.yaml'": "'config/unified_config/unified_config.yaml'",
                'config/unified_config/unified_config.yaml': 'config/unified_config/unified_config.yaml',
            }
            
            for old_ref, new_ref in config_file_updates.items():
                if old_ref in content:
                    content = content.replace(old_ref, new_ref)
                    modified = True
                    logger.info(f"Actualizada referencia en {file_path}: {old_ref} -> {new_ref}")
            
            # Actualizar rutas hardcodeadas
            hardcoded_path_pattern = r''
            if re.search(hardcoded_path_pattern, content):
                # Reemplazar rutas absolutas con rutas relativas o configuración
                content = re.sub(hardcoded_path_pattern, '', content)
                modified = True
                logger.info(f"Removidas rutas hardcodeadas en {file_path}")
            
            if modified:
                self.create_backup(file_path)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.migration_report['files_modified'].append(str(file_path))
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Error actualizando {file_path}: {e}"
            logger.error(error_msg)
            self.migration_report['errors'].append(error_msg)
            return False
    
    def update_deep_learning_configs(self) -> None:
        """Actualiza configuraciones de deep learning para usar el sistema unificado"""
        logger.info("Actualizando configuraciones de deep learning...")
        
        dl_config_dir = self.project_root / "deep_learning" / "config"
        if not dl_config_dir.exists():
            return
        
        # Archivos de configuración de deep learning a actualizar
        dl_files = [
            dl_config_dir / "config_manager.py",
            dl_config_dir / "hyperparameter_config.py",
            dl_config_dir / "training_configs.py"
        ]
        
        for dl_file in dl_files:
            if dl_file.exists():
                self.update_python_imports(dl_file)
                
                # Actualizar referencias específicas de deep learning
                try:
                    with open(dl_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Actualizar rutas hardcodeadas específicas de deep learning
                    dl_updates = {
                        '"deep_learning/configs"': 'get_unified_config().get_absolute_path("deep_learning/configs")',
                        '"deep_learning/models"': 'get_unified_config().get_absolute_path("deep_learning/models")',
                        '"uploads/Muestras NIST FADB"': 'get_unified_config().get_absolute_path("uploads/Muestras NIST FADB")',
                    }
                    
                    for old_path, new_path in dl_updates.items():
                        if old_path in content:
                            content = content.replace(old_path, new_path)
                            logger.info(f"Actualizada ruta DL en {dl_file}: {old_path} -> {new_path}")
                    
                    if content != original_content:
                        self.create_backup(dl_file)
                        with open(dl_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        self.migration_report['files_modified'].append(str(dl_file))
                
                except Exception as e:
                    error_msg = f"Error actualizando configuración DL {dl_file}: {e}"
                    logger.error(error_msg)
                    self.migration_report['errors'].append(error_msg)
    
    def scan_and_update_python_files(self) -> None:
        """Escanea y actualiza todos los archivos Python del proyecto"""
        logger.info("Escaneando archivos Python para actualizar referencias...")
        
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Excluir directorios específicos
            dirs[:] = [d for d in dirs if not any(re.search(pattern, f"{root}/{d}") for pattern in self.exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    if not self.should_exclude_file(file_path):
                        python_files.append(file_path)
        
        logger.info(f"Encontrados {len(python_files)} archivos Python para procesar")
        
        modified_count = 0
        for py_file in python_files:
            self.migration_report['files_processed'].append(str(py_file))
            
            if self.update_python_imports(py_file):
                modified_count += 1
        
        logger.info(f"Modificados {modified_count} archivos Python")
        self.migration_report['summary']['python_files_modified'] = modified_count
    
    def update_gui_files(self) -> None:
        """Actualiza archivos específicos de GUI"""
        logger.info("Actualizando archivos de GUI...")
        
        gui_files = [
            self.project_root / "gui" / "main_window.py",
            self.project_root / "gui" / "advanced_config_panel.py"
        ]
        
        for gui_file in gui_files:
            if gui_file.exists():
                self.update_python_imports(gui_file)
                
                # Actualizar referencias específicas de GUI
                try:
                    with open(gui_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Actualizar referencias a configuración de GUI
                    gui_updates = {
                        'self.config.gui': 'get_unified_config().gui',
                        'config.gui': 'get_unified_config().gui',
                    }
                    
                    for old_ref, new_ref in gui_updates.items():
                        if old_ref in content:
                            content = content.replace(old_ref, new_ref)
                            logger.info(f"Actualizada referencia GUI en {gui_file}: {old_ref} -> {new_ref}")
                    
                    if content != original_content:
                        self.create_backup(gui_file)
                        with open(gui_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        self.migration_report['files_modified'].append(str(gui_file))
                
                except Exception as e:
                    error_msg = f"Error actualizando GUI {gui_file}: {e}"
                    logger.error(error_msg)
                    self.migration_report['errors'].append(error_msg)
    
    def create_compatibility_wrapper(self) -> None:
        """Crea un wrapper de compatibilidad para utils/config.py"""
        logger.info("Creando wrapper de compatibilidad...")
        
        utils_config_path = self.project_root / "utils" / "config.py"
        
        if utils_config_path.exists():
            self.create_backup(utils_config_path)
            
            compatibility_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper de Compatibilidad - utils/config.py
==========================================

Este archivo mantiene compatibilidad con código legacy que importa
configuraciones desde utils.config. Redirige todas las llamadas
al nuevo sistema de configuración unificado.

DEPRECADO: Use config.unified_config directamente en código nuevo.

Autor: SIGeC-BalisticaTeam
"""

import warnings
from config.unified_config import (
    get_unified_config,
    get_database_config,
    get_image_processing_config,
    get_matching_config,
    get_gui_config,
    get_logging_config,
    get_deep_learning_config,
    get_nist_config,
    UnifiedConfig,
    DatabaseConfig,
    ImageProcessingConfig,
    MatchingConfig,
    GUIConfig,
    LoggingConfig,
    DeepLearningConfig,
    NISTConfig
)

# Emitir advertencia de deprecación
warnings.warn(
    "utils.config está deprecado. Use config.unified_config directamente.",
    DeprecationWarning,
    stacklevel=2
)

# Alias para compatibilidad
Config = get_unified_config

# Funciones de compatibilidad
def get_config():
    """Función de compatibilidad"""
    return get_unified_config()

# Exportar clases para compatibilidad
__all__ = [
    'Config',
    'get_config',
    'UnifiedConfig',
    'DatabaseConfig',
    'ImageProcessingConfig',
    'MatchingConfig',
    'GUIConfig',
    'LoggingConfig',
    'DeepLearningConfig',
    'NISTConfig',
    'get_unified_config',
    'get_database_config',
    'get_image_processing_config',
    'get_matching_config',
    'get_gui_config',
    'get_logging_config',
    'get_deep_learning_config',
    'get_nist_config'
]
'''
            
            with open(utils_config_path, 'w', encoding='utf-8') as f:
                f.write(compatibility_content)
            
            logger.info("Wrapper de compatibilidad creado en utils/config.py")
            self.migration_report['files_modified'].append(str(utils_config_path))
    
    def generate_migration_report(self) -> None:
        """Genera reporte de migración"""
        self.migration_report['end_time'] = datetime.now().isoformat()
        self.migration_report['summary'].update({
            'total_files_processed': len(self.migration_report['files_processed']),
            'total_files_modified': len(self.migration_report['files_modified']),
            'total_errors': len(self.migration_report['errors']),
            'total_warnings': len(self.migration_report['warnings']),
            'backup_directory': str(self.backup_dir)
        })
        
        report_file = self.project_root / "migration_report.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.migration_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Reporte de migración guardado en: {report_file}")
            
            # Mostrar resumen en consola
            print("\n" + "="*60)
            print("REPORTE DE MIGRACIÓN A CONFIGURACIÓN UNIFICADA")
            print("="*60)
            print(f"Archivos procesados: {self.migration_report['summary']['total_files_processed']}")
            print(f"Archivos modificados: {self.migration_report['summary']['total_files_modified']}")
            print(f"Errores: {self.migration_report['summary']['total_errors']}")
            print(f"Advertencias: {self.migration_report['summary']['total_warnings']}")
            print(f"Respaldos en: {self.backup_dir}")
            
            if self.migration_report['errors']:
                print("\nERRORES:")
                for error in self.migration_report['errors']:
                    print(f"  - {error}")
            
            if self.migration_report['warnings']:
                print("\nADVERTENCIAS:")
                for warning in self.migration_report['warnings']:
                    print(f"  - {warning}")
            
            print("\n✅ Migración completada. Revise el reporte detallado en migration_report.json")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
    
    def run_migration(self) -> None:
        """Ejecuta la migración completa"""
        logger.info("Iniciando migración a configuración unificada...")
        
        try:
            # 1. Migrar archivos de configuración legacy
            self.migrate_legacy_config_files()
            
            # 2. Actualizar configuraciones de deep learning
            self.update_deep_learning_configs()
            
            # 3. Actualizar archivos de GUI
            self.update_gui_files()
            
            # 4. Escanear y actualizar todos los archivos Python
            self.scan_and_update_python_files()
            
            # 5. Crear wrapper de compatibilidad
            self.create_compatibility_wrapper()
            
            # 6. Generar reporte
            self.generate_migration_report()
            
            logger.info("Migración completada exitosamente")
            
        except Exception as e:
            logger.error(f"Error durante la migración: {e}")
            self.migration_report['errors'].append(f"Error general: {e}")
            self.generate_migration_report()
            raise

def main():
    """Función principal"""
    print("Script de Migración a Configuración Unificada - SIGeC-Balistica")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    project_root = Path.cwd()
    if not (project_root / "main.py").exists():
        print("❌ Error: Ejecute este script desde el directorio raíz del proyecto SIGeC-Balistica")
        sys.exit(1)
    
    # Confirmar migración
    response = input("\n¿Desea proceder con la migración? (s/N): ").lower().strip()
    if response not in ['s', 'si', 'sí', 'y', 'yes']:
        print("Migración cancelada.")
        sys.exit(0)
    
    try:
        migrator = ConfigMigrator(str(project_root))
        migrator.run_migration()
        
    except KeyboardInterrupt:
        print("\n❌ Migración interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la migración: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()