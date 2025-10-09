#!/usr/bin/env python3
"""
Script de Limpieza de Archivos de Backup
========================================

Implementa política de retención automática para archivos de backup
del sistema SIGeC-Balisticar.

Funcionalidades:
- Política de retención configurable
- Compresión de backups antiguos
- Limpieza automática por fecha/cantidad
- Preservación de backups críticos

Uso:
    python cleanup_backup_files.py [--dry-run] [--keep=5] [--compress]
"""

import os
import gzip
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import yaml

class BackupCleaner:
    """Limpiador de archivos de backup con política de retención"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.backup_dirs = [
            self.project_root / "config" / "backups",
            self.project_root / "logs" / "backups",
            self.project_root / "data" / "backups",
        ]
        
        # Configuración por defecto
        self.default_policy = {
            'keep_recent': 5,           # Mantener últimos N archivos
            'keep_days': 30,            # Mantener archivos de últimos N días
            'compress_after_days': 7,   # Comprimir archivos después de N días
            'delete_after_days': 90,    # Eliminar archivos después de N días
            'preserve_patterns': [      # Patrones de archivos a preservar
                '*_critical_*',
                '*_production_*',
                '*_release_*'
            ]
        }
        
        self.stats = {
            'files_found': 0,
            'files_compressed': 0,
            'files_deleted': 0,
            'space_freed': 0
        }
    
    def load_retention_policy(self) -> Dict:
        """Carga política de retención desde configuración"""
        config_file = self.project_root / "config" / "backup_policy.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    policy = yaml.safe_load(f)
                    return {**self.default_policy, **policy}
            except Exception as e:
                print(f"⚠️  Error cargando política: {e}")
        
        return self.default_policy
    
    def create_default_policy_file(self):
        """Crea archivo de política por defecto"""
        policy_file = self.project_root / "config" / "backup_policy.yaml"
        
        if not policy_file.exists():
            policy_content = """# Política de Retención de Backups
# ================================

# Mantener últimos N archivos por directorio
keep_recent: 5

# Mantener archivos de últimos N días
keep_days: 30

# Comprimir archivos después de N días
compress_after_days: 7

# Eliminar archivos después de N días
delete_after_days: 90

# Patrones de archivos a preservar siempre
preserve_patterns:
  - "*_critical_*"
  - "*_production_*" 
  - "*_release_*"
  - "*_migration_*"

# Configuración por tipo de archivo
file_types:
  config:
    keep_recent: 10
    compress_after_days: 3
  logs:
    keep_recent: 7
    compress_after_days: 1
  data:
    keep_recent: 3
    compress_after_days: 14
"""
            
            policy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(policy_file, 'w', encoding='utf-8') as f:
                f.write(policy_content)
            
            print(f"✅ Política creada: {policy_file}")
    
    def get_file_info(self, file_path: Path) -> Dict:
        """Obtiene información detallada de un archivo"""
        stat = file_path.stat()
        
        return {
            'path': file_path,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'age_days': (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days,
            'is_compressed': file_path.suffix.lower() in ['.gz', '.zip', '.bz2'],
            'should_preserve': self.should_preserve_file(file_path)
        }
    
    def should_preserve_file(self, file_path: Path) -> bool:
        """Verifica si un archivo debe preservarse"""
        policy = self.load_retention_policy()
        
        for pattern in policy['preserve_patterns']:
            if file_path.match(pattern):
                return True
        
        return False
    
    def categorize_files(self, backup_dir: Path) -> Dict[str, List[Dict]]:
        """Categoriza archivos de backup por tipo y fecha"""
        if not backup_dir.exists():
            return {}
        
        files_by_category = {
            'recent': [],      # Archivos recientes a mantener
            'compress': [],    # Archivos a comprimir
            'delete': [],      # Archivos a eliminar
            'preserve': []     # Archivos a preservar siempre
        }
        
        policy = self.load_retention_policy()
        
        # Obtener todos los archivos de backup
        backup_files = []
        for pattern in ['*.yaml', '*.json', '*.log', '*.backup', '*.bak']:
            backup_files.extend(backup_dir.glob(pattern))
        
        # Procesar cada archivo
        for file_path in backup_files:
            if file_path.is_file():
                file_info = self.get_file_info(file_path)
                self.stats['files_found'] += 1
                
                # Archivos a preservar siempre
                if file_info['should_preserve']:
                    files_by_category['preserve'].append(file_info)
                    continue
                
                # Categorizar por edad
                age_days = file_info['age_days']
                
                if age_days <= policy['keep_days']:
                    files_by_category['recent'].append(file_info)
                elif age_days <= policy['compress_after_days'] and not file_info['is_compressed']:
                    files_by_category['compress'].append(file_info)
                elif age_days >= policy['delete_after_days']:
                    files_by_category['delete'].append(file_info)
                else:
                    files_by_category['recent'].append(file_info)
        
        # Ordenar por fecha (más recientes primero)
        for category in files_by_category.values():
            category.sort(key=lambda x: x['modified'], reverse=True)
        
        # Aplicar límite de archivos recientes
        keep_recent = policy['keep_recent']
        if len(files_by_category['recent']) > keep_recent:
            excess_files = files_by_category['recent'][keep_recent:]
            files_by_category['recent'] = files_by_category['recent'][:keep_recent]
            
            # Mover exceso a comprimir o eliminar según edad
            for file_info in excess_files:
                if file_info['age_days'] >= policy['delete_after_days']:
                    files_by_category['delete'].append(file_info)
                else:
                    files_by_category['compress'].append(file_info)
        
        return files_by_category
    
    def compress_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Comprime un archivo usando gzip"""
        if file_path.suffix.lower() in ['.gz', '.zip', '.bz2']:
            return False  # Ya está comprimido
        
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        if dry_run:
            print(f"  [DRY-RUN] Comprimir: {file_path.name} → {compressed_path.name}")
            return True
        
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Preservar timestamps
            stat = file_path.stat()
            os.utime(compressed_path, (stat.st_atime, stat.st_mtime))
            
            # Eliminar archivo original
            original_size = file_path.stat().st_size
            file_path.unlink()
            
            compressed_size = compressed_path.stat().st_size
            space_saved = original_size - compressed_size
            self.stats['space_freed'] += space_saved
            self.stats['files_compressed'] += 1
            
            print(f"  ✅ Comprimido: {file_path.name} (ahorró {self.format_size(space_saved)})")
            return True
            
        except Exception as e:
            print(f"  ❌ Error comprimiendo {file_path.name}: {e}")
            return False
    
    def delete_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """Elimina un archivo de backup"""
        if dry_run:
            print(f"  [DRY-RUN] Eliminar: {file_path.name}")
            return True
        
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            
            self.stats['space_freed'] += file_size
            self.stats['files_deleted'] += 1
            
            print(f"  🗑️  Eliminado: {file_path.name} ({self.format_size(file_size)})")
            return True
            
        except Exception as e:
            print(f"  ❌ Error eliminando {file_path.name}: {e}")
            return False
    
    def format_size(self, size_bytes: int) -> str:
        """Formatea tamaño en bytes a formato legible"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def process_backup_directory(self, backup_dir: Path, dry_run: bool = False):
        """Procesa un directorio de backups"""
        if not backup_dir.exists():
            return
        
        print(f"\n📁 Procesando: {backup_dir.relative_to(self.project_root)}")
        
        # Categorizar archivos
        categorized = self.categorize_files(backup_dir)
        
        # Mostrar resumen
        print(f"  📊 Archivos encontrados:")
        print(f"     Recientes (mantener): {len(categorized['recent'])}")
        print(f"     A comprimir: {len(categorized['compress'])}")
        print(f"     A eliminar: {len(categorized['delete'])}")
        print(f"     Preservados: {len(categorized['preserve'])}")
        
        # Comprimir archivos
        if categorized['compress']:
            print(f"  🗜️  Comprimiendo {len(categorized['compress'])} archivos:")
            for file_info in categorized['compress']:
                self.compress_file(file_info['path'], dry_run)
        
        # Eliminar archivos
        if categorized['delete']:
            print(f"  🗑️  Eliminando {len(categorized['delete'])} archivos:")
            for file_info in categorized['delete']:
                self.delete_file(file_info['path'], dry_run)
    
    def generate_report(self) -> Dict:
        """Genera reporte de limpieza"""
        return {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'statistics': self.stats,
            'policy': self.load_retention_policy()
        }
    
    def save_report(self, report: Dict):
        """Guarda reporte de limpieza"""
        report_dir = self.project_root / "logs"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"backup_cleanup_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte guardado: {report_path}")
    
    def clean_all_backups(self, dry_run: bool = False):
        """Limpia todos los directorios de backup"""
        print("🧹 LIMPIEZA DE ARCHIVOS DE BACKUP")
        print("=" * 50)
        print(f"📋 Modo: {'DRY-RUN (solo análisis)' if dry_run else 'LIMPIEZA ACTIVA'}")
        
        # Crear política por defecto si no existe
        if not dry_run:
            self.create_default_policy_file()
        
        # Procesar cada directorio
        for backup_dir in self.backup_dirs:
            self.process_backup_directory(backup_dir, dry_run)
        
        # Mostrar resumen final
        print(f"\n📊 RESUMEN FINAL:")
        print(f"  Archivos encontrados: {self.stats['files_found']}")
        print(f"  Archivos comprimidos: {self.stats['files_compressed']}")
        print(f"  Archivos eliminados: {self.stats['files_deleted']}")
        print(f"  Espacio liberado: {self.format_size(self.stats['space_freed'])}")
        
        # Generar reporte
        if not dry_run and (self.stats['files_compressed'] > 0 or self.stats['files_deleted'] > 0):
            report = self.generate_report()
            self.save_report(report)
        
        if dry_run:
            print(f"\n💡 Para aplicar cambios, ejecute sin --dry-run")
        elif self.stats['files_compressed'] > 0 or self.stats['files_deleted'] > 0:
            print(f"\n✅ Limpieza completada exitosamente!")
        else:
            print(f"\n✅ No se requiere limpieza")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Limpieza de archivos de backup")
    parser.add_argument("--dry-run", action="store_true",
                       help="Solo analizar, no modificar archivos")
    parser.add_argument("--keep", type=int, default=5,
                       help="Número de archivos recientes a mantener")
    parser.add_argument("--compress", action="store_true", default=True,
                       help="Comprimir archivos antiguos (habilitado por defecto)")
    parser.add_argument("--project-root", type=str,
                       help="Ruta raíz del proyecto")
    
    args = parser.parse_args()
    
    # Inicializar limpiador
    cleaner = BackupCleaner(args.project_root)
    
    # Ajustar política si se especificó --keep
    if args.keep != 5:
        cleaner.default_policy['keep_recent'] = args.keep
    
    # Ejecutar limpieza
    cleaner.clean_all_backups(dry_run=args.dry_run)

if __name__ == "__main__":
    main()