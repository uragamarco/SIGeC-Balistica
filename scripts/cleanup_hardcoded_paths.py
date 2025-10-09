#!/usr/bin/env python3
"""
Script de Limpieza de Rutas Hardcodeadas
=======================================

Este script automatiza la refactorización de rutas absolutas hardcodeadas
identificadas en el análisis del repositorio SIGeC-Balisticar.

Funcionalidades:
- Detecta rutas hardcodeadas automáticamente
- Reemplaza con referencias a configuración centralizada
- Genera backup antes de modificaciones
- Valida cambios realizados

Uso:
    python cleanup_hardcoded_paths.py [--dry-run] [--backup]
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import yaml

class HardcodedPathCleaner:
    """Limpiador de rutas hardcodeadas"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.backup_dir = self.project_root / "config" / "backups" / "path_cleanup"
        self.config_file = self.project_root / "config" / "unified_config.yaml"
        
        # Patrones de rutas hardcodeadas a detectar
        self.hardcoded_patterns = [
            r'/home/marco/SIGeC-Balistica(?!/config/backups)',  # Ruta antigua
            r'/home/marco/SIGeC-Balisticar(?!/config/backups)', # Ruta actual
            r'localhost:5000',
            r'127\.0\.0\.1:?\d*',
            r'http://localhost:\d+',
        ]
        
        # Archivos a excluir del procesamiento
        self.exclude_patterns = [
            r'.*\.git/.*',
            r'.*/__pycache__/.*',
            r'.*\.pyc$',
            r'.*config/backups/.*',
            r'.*\.md$',  # Documentación puede contener ejemplos
            r'.*cleanup_hardcoded_paths\.py$',  # Este script
            r'.*/venv/.*',
            r'.*/.venv/.*',
            r'.*/env/.*',
            r'.*/.env/.*',
            r'.*/site-packages/.*',
        ]
        
        self.changes_made = []
        
    def load_config(self) -> Dict:
        """Carga la configuración unificada"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Archivo de configuración no encontrado: {self.config_file}")
            return {}
    
    def should_exclude_file(self, file_path: Path) -> bool:
        """Verifica si un archivo debe ser excluido"""
        file_str = str(file_path)
        return any(re.match(pattern, file_str) for pattern in self.exclude_patterns)
    
    def find_hardcoded_paths(self, content: str) -> List[Tuple[str, int]]:
        """Encuentra rutas hardcodeadas en el contenido"""
        findings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern in self.hardcoded_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    findings.append((match.group(), line_num))
        
        return findings
    
    def generate_replacement(self, hardcoded_path: str) -> str:
        """Genera el reemplazo apropiado para una ruta hardcodeada"""
        
        # Rutas de proyecto
        if '/home/marco/SIGeC-Balistica' in hardcoded_path or '/home/marco/SIGeC-Balisticar' in hardcoded_path:
            return "get_project_root()"
        
        # URLs y puertos
        if 'localhost:5000' in hardcoded_path:
            return "get_config_value('api.host', 'localhost') + ':' + str(get_config_value('api.port', 5000))"
        
        if '127.0.0.1' in hardcoded_path:
            return "get_config_value('api.host', '127.0.0.1')"
        
        if 'http://localhost:' in hardcoded_path:
            port_match = re.search(r':(\d+)', hardcoded_path)
            if port_match:
                port = port_match.group(1)
                return f"f\"http://{{get_config_value('api.host', 'localhost')}}:{{get_config_value('api.port', {port})}}\""
        
        return hardcoded_path  # Fallback
    
    def add_config_imports(self, content: str, file_path: Path) -> str:
        """Añade imports necesarios para configuración"""
        
        # Verificar si ya tiene los imports
        if 'get_project_root' in content or 'get_config_value' in content:
            return content
        
        # Determinar el import apropiado basado en la ubicación del archivo
        relative_path = file_path.relative_to(self.project_root)
        depth = len(relative_path.parts) - 1
        
        if depth == 0:
            import_line = "from config.config_manager import get_project_root, get_config_value"
        else:
            dots = '.' * depth
            import_line = f"from {dots}config.config_manager import get_project_root, get_config_value"
        
        # Buscar donde insertar el import
        lines = content.split('\n')
        insert_pos = 0
        
        # Buscar después de imports existentes
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_pos = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        lines.insert(insert_pos, import_line)
        return '\n'.join(lines)
    
    def process_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Procesa un archivo individual"""
        
        if self.should_exclude_file(file_path):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return False
        
        # Buscar rutas hardcodeadas
        findings = self.find_hardcoded_paths(original_content)
        if not findings:
            return False
        
        print(f"\n📁 Procesando: {file_path.relative_to(self.project_root)}")
        
        modified_content = original_content
        changes_in_file = []
        
        # Procesar cada hallazgo
        for hardcoded_path, line_num in findings:
            replacement = self.generate_replacement(hardcoded_path)
            
            if replacement != hardcoded_path:
                print(f"  Línea {line_num}: '{hardcoded_path}' → '{replacement}'")
                modified_content = modified_content.replace(hardcoded_path, replacement)
                changes_in_file.append({
                    'line': line_num,
                    'original': hardcoded_path,
                    'replacement': replacement
                })
        
        # Añadir imports si es necesario
        if changes_in_file and file_path.suffix == '.py':
            modified_content = self.add_config_imports(modified_content, file_path)
        
        # Aplicar cambios si no es dry-run
        if not dry_run and modified_content != original_content:
            # Crear backup
            self.create_backup(file_path, original_content)
            
            # Escribir archivo modificado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            self.changes_made.append({
                'file': str(file_path.relative_to(self.project_root)),
                'changes': changes_in_file
            })
        
        return len(changes_in_file) > 0
    
    def create_backup(self, file_path: Path, content: str):
        """Crea backup de un archivo"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relative_path = file_path.relative_to(self.project_root)
        backup_name = f"{relative_path.as_posix().replace('/', '_')}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  💾 Backup creado: {backup_name}")
    
    def scan_project(self, dry_run: bool = False) -> Dict:
        """Escanea todo el proyecto buscando rutas hardcodeadas"""
        
        print(f"🔍 Escaneando proyecto: {self.project_root}")
        print(f"📋 Modo: {'DRY-RUN (solo análisis)' if dry_run else 'MODIFICACIÓN'}")
        
        files_processed = 0
        files_modified = 0
        
        # Buscar archivos Python
        for file_path in self.project_root.rglob("*.py"):
            if self.process_file(file_path, dry_run):
                files_modified += 1
            files_processed += 1
        
        # Resumen
        print(f"\n📊 RESUMEN:")
        print(f"  Archivos procesados: {files_processed}")
        print(f"  Archivos con cambios: {files_modified}")
        
        if not dry_run and self.changes_made:
            self.generate_change_report()
        
        return {
            'files_processed': files_processed,
            'files_modified': files_modified,
            'changes': self.changes_made
        }
    
    def generate_change_report(self):
        """Genera reporte de cambios realizados"""
        report_path = self.backup_dir / f"change_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'project_root': str(self.project_root),
                'changes': self.changes_made
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte de cambios: {report_path}")
    
    def create_config_manager(self):
        """Crea el módulo config_manager si no existe"""
        config_manager_path = self.project_root / "config" / "config_manager.py"
        
        if config_manager_path.exists():
            print(f"✅ config_manager.py ya existe")
            return
        
        config_manager_content = '''"""
Gestor de Configuración Centralizada
===================================

Proporciona funciones para acceder a la configuración unificada
y obtener rutas del proyecto de forma portable.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional

_config_cache = None
_project_root = None

def get_project_root() -> str:
    """Obtiene la ruta raíz del proyecto de forma portable"""
    global _project_root
    
    if _project_root is None:
        # Buscar desde el directorio actual hacia arriba
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "config" / "unified_config.yaml").exists():
                _project_root = str(current)
                break
            current = current.parent
        else:
            # Fallback al directorio padre de config
            _project_root = str(Path(__file__).parent.parent)
    
    return _project_root

def load_config() -> dict:
    """Carga la configuración unificada"""
    global _config_cache
    
    if _config_cache is None:
        config_path = Path(get_project_root()) / "config" / "unified_config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                _config_cache = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️  Configuración no encontrada: {config_path}")
            _config_cache = {}
    
    return _config_cache

def get_config_value(key: str, default: Any = None) -> Any:
    """Obtiene un valor de configuración usando notación de punto"""
    config = load_config()
    
    # Navegar por claves anidadas (ej: "api.host")
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def reload_config():
    """Recarga la configuración desde disco"""
    global _config_cache
    _config_cache = None
'''
        
        config_manager_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_manager_path, 'w', encoding='utf-8') as f:
            f.write(config_manager_content)
        
        print(f"✅ Creado: {config_manager_path}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Limpieza de rutas hardcodeadas")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Solo analizar, no modificar archivos")
    parser.add_argument("--backup", action="store_true", default=True,
                       help="Crear backups (habilitado por defecto)")
    parser.add_argument("--project-root", type=str,
                       help="Ruta raíz del proyecto")
    
    args = parser.parse_args()
    
    # Inicializar limpiador
    cleaner = HardcodedPathCleaner(args.project_root)
    
    print("🧹 LIMPIEZA DE RUTAS HARDCODEADAS")
    print("=" * 50)
    
    # Crear config_manager si no existe
    if not args.dry_run:
        cleaner.create_config_manager()
    
    # Escanear proyecto
    results = cleaner.scan_project(dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"\n💡 Para aplicar cambios, ejecute sin --dry-run")
    elif results['files_modified'] > 0:
        print(f"\n✅ Limpieza completada exitosamente!")
        print(f"   Backups disponibles en: {cleaner.backup_dir}")
    else:
        print(f"\n✅ No se encontraron rutas hardcodeadas para limpiar")

if __name__ == "__main__":
    main()