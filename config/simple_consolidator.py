#!/usr/bin/env python3
"""
Consolidador simplificado de configuraciones para SIGeC-Balistica.
Versión robusta que maneja la consolidación sin errores complejos.
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class SimpleConfigConsolidator:
    """Consolidador simplificado de configuraciones."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "config" / "backups" / "simple_migration"
        self.unified_config_path = self.root_path / "config" / "unified_config_consolidated.yaml"
        
    def consolidate(self) -> Dict[str, Any]:
        """Ejecuta la consolidación simplificada."""
        print("=== CONSOLIDACIÓN SIMPLIFICADA DE CONFIGURACIONES ===")
        
        # Crear directorio de backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Cargar configuración base
        base_config = self._load_base_config()
        
        # Buscar y consolidar configuraciones adicionales
        additional_configs = self._find_additional_configs()
        
        # Crear backups
        self._create_backups(additional_configs, backup_path)
        
        # Fusionar configuraciones
        consolidated = self._merge_configs(base_config, additional_configs)
        
        # Guardar configuración consolidada
        self._save_consolidated_config(consolidated)
        
        print(f"✅ Consolidación completada. Backup en: {backup_path}")
        print(f"✅ Configuración consolidada guardada en: {self.unified_config_path}")
        
        return {
            'status': 'success',
            'backup_path': str(backup_path),
            'consolidated_path': str(self.unified_config_path),
            'configs_processed': len(additional_configs) + 1
        }
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Carga la configuración base unificada."""
        base_path = self.root_path / "config" / "unified_config.yaml"
        
        if base_path.exists():
            try:
                with open(base_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"⚠️ Error cargando config base: {e}")
                return {}
        
        return {}
    
    def _find_additional_configs(self) -> List[Dict[str, Any]]:
        """Encuentra configuraciones adicionales para consolidar."""
        configs = []
        
        # Buscar archivos de configuración
        config_patterns = [
            "config/*.yaml",
            "config/*.yml", 
            "config/*.json",
            "*/config/*.yaml",
            "*/config/*.yml",
            "*/config/*.json",
            "tests/config.yaml",
            "deep_learning/config/*.yaml"
        ]
        
        for pattern in config_patterns:
            for config_file in self.root_path.glob(pattern):
                if config_file.name != "unified_config.yaml" and config_file.name != "unified_config_consolidated.yaml":
                    config_data = self._load_config_file(config_file)
                    if config_data:
                        configs.append({
                            'path': config_file,
                            'data': config_data
                        })
        
        return configs
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Carga un archivo de configuración individual."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
        except Exception as e:
            print(f"⚠️ Error cargando {file_path}: {e}")
        
        return {}
    
    def _create_backups(self, configs: List[Dict[str, Any]], backup_path: Path):
        """Crea backups de las configuraciones."""
        for config in configs:
            try:
                src_path = config['path']
                dst_path = backup_path / src_path.name
                shutil.copy2(src_path, dst_path)
                print(f"📁 Backup creado: {dst_path}")
            except Exception as e:
                print(f"⚠️ Error creando backup de {config['path']}: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], additional_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fusiona todas las configuraciones de manera inteligente."""
        consolidated = base_config.copy()
        
        for config in additional_configs:
            config_data = config['data']
            config_path = config['path']
            
            print(f"🔄 Procesando: {config_path}")
            
            # Fusión inteligente por secciones
            for key, value in config_data.items():
                if key not in consolidated:
                    consolidated[key] = value
                elif isinstance(consolidated[key], dict) and isinstance(value, dict):
                    # Fusionar diccionarios recursivamente
                    consolidated[key] = self._merge_dicts(consolidated[key], value)
                else:
                    # Mantener valor existente pero reportar conflicto
                    print(f"⚠️ Conflicto en '{key}': manteniendo valor existente")
        
        return consolidated
    
    def _merge_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Fusiona dos diccionarios recursivamente."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key not in result:
                result[key] = value
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            # Si hay conflicto, mantener el valor original
        
        return result
    
    def _save_consolidated_config(self, config: Dict[str, Any]):
        """Guarda la configuración consolidada."""
        try:
            with open(self.unified_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"💾 Configuración consolidada guardada")
        except Exception as e:
            print(f"❌ Error guardando configuración consolidada: {e}")

def consolidate_system_simple(root_path: str = None) -> Dict[str, Any]:
    """Función de conveniencia para consolidar configuraciones."""
    if root_path is None:
        root_path = os.getcwd()
    
    consolidator = SimpleConfigConsolidator(root_path)
    return consolidator.consolidate()

if __name__ == "__main__":
    result = consolidate_system_simple("/home/marco/SIGeC-Balistica")
    print(f"\n🎉 Resultado: {result}")