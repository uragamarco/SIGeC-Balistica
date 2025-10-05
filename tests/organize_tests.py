#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Organización de Archivos de Prueba
Sistema Balístico Forense SIGeC-Balistica

Consolida archivos de prueba duplicados y crea una estructura organizada.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

class TestOrganizer:
    """Organizador de archivos de prueba"""
    
    def __init__(self, root_dir: str = "/home/marco/SIGeC-Balistica"):
        self.root_dir = Path(root_dir)
        self.tests_dir = self.root_dir / "tests"
        self.backup_dir = self.root_dir / "test_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio de respaldo
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_test_structure(self):
        """Crea la estructura organizada de tests"""
        
        # Estructura objetivo
        structure = {
            "unit": "Pruebas unitarias de módulos individuales",
            "integration": "Pruebas de integración entre módulos",
            "performance": "Pruebas de rendimiento y optimización",
            "gui": "Pruebas de interfaz gráfica",
            "data": "Datos de prueba y fixtures",
            "reports": "Reportes de ejecución de pruebas",
            "legacy": "Archivos de prueba obsoletos (para revisión)"
        }
        
        for folder, description in structure.items():
            folder_path = self.tests_dir / folder
            folder_path.mkdir(exist_ok=True)
            
            # Crear __init__.py
            init_file = folder_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""{description}"""\n')
                
        print("✅ Estructura de directorios creada")
        
    def move_root_tests(self):
        """Mueve archivos de prueba del directorio raíz a tests/"""
        
        # Mapeo de archivos a categorías
        file_mapping = {
            "unit": [
                "simple_image_test.py",
                "advanced_module_test.py",
                "test_image_processing.py",
                "test_memory_optimized.py"
            ],
            "integration": [
                "test_backend_integration.py",
                "test_complete_integration.py"
            ],
            "gui": [
                "test_roi_visualization.py",
                "test_simple_roi_viz.py",
                "test_visualization_features.py"
            ]
        }
        
        moved_files = []
        
        for category, files in file_mapping.items():
            for filename in files:
                source = self.root_dir / filename
                if source.exists():
                    # Crear respaldo
                    backup_file = self.backup_dir / filename
                    shutil.copy2(source, backup_file)
                    
                    # Mover a la nueva ubicación
                    destination = self.tests_dir / category / filename
                    shutil.move(source, destination)
                    moved_files.append(f"{filename} → tests/{category}/")
                    
        print(f"✅ Movidos {len(moved_files)} archivos de prueba:")
        for file_move in moved_files:
            print(f"   {file_move}")
            
    def consolidate_integration_legacy(self):
        """Consolida integration_legacy en legacy"""
        
        legacy_source = self.tests_dir / "integration_legacy"
        legacy_dest = self.tests_dir / "legacy"
        
        if legacy_source.exists():
            # Mover archivos individuales
            for file_path in legacy_source.glob("*.py"):
                destination = legacy_dest / file_path.name
                shutil.move(file_path, destination)
                
            # Remover directorio vacío
            if not any(legacy_source.iterdir()):
                legacy_source.rmdir()
                
            print("✅ Archivos legacy consolidados")
            
    def clean_empty_directories(self):
        """Remueve directorios vacíos"""
        
        removed_dirs = []
        
        def remove_empty_dirs(path):
            if path.is_dir():
                # Procesar subdirectorios primero
                for subdir in path.iterdir():
                    if subdir.is_dir():
                        remove_empty_dirs(subdir)
                
                # Si el directorio está vacío, removerlo
                try:
                    if not any(path.iterdir()):
                        path.rmdir()
                        removed_dirs.append(str(path.relative_to(self.root_dir)))
                except OSError:
                    pass  # Directorio no vacío o sin permisos
                    
        # Buscar en todo el proyecto
        for item in self.root_dir.rglob("*"):
            if item.is_dir() and item.name not in ['.git', '.vscode', '__pycache__']:
                remove_empty_dirs(item)
                
        if removed_dirs:
            print(f"✅ Removidos {len(removed_dirs)} directorios vacíos:")
            for dir_name in removed_dirs:
                print(f"   {dir_name}")
        else:
            print("✅ No se encontraron directorios vacíos")
            
    def create_pytest_config(self):
        """Crea configuración de pytest"""
        
        pytest_ini = self.root_dir / "pytest.ini"
        config_content = """[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
    --durations=10
markers =
    unit: Pruebas unitarias
    integration: Pruebas de integración
    performance: Pruebas de rendimiento
    gui: Pruebas de interfaz gráfica
    slow: Pruebas que tardan más de 30 segundos
    nist: Pruebas con estándares NIST
"""
        
        pytest_ini.write_text(config_content)
        print("✅ Configuración pytest.ini creada")
        
    def create_test_readme(self):
        """Crea documentación de la estructura de tests"""
        
        readme_content = """# Estructura de Pruebas - SIGeC-Balistica

## Organización de Directorios

### 📁 `unit/`
Pruebas unitarias de módulos individuales:
- `simple_image_test.py` - Pruebas básicas de procesamiento de imágenes
- `advanced_module_test.py` - Pruebas avanzadas de módulos
- `test_image_processing.py` - Pruebas específicas de procesamiento
- `test_memory_optimized.py` - Pruebas de optimización de memoria

### 📁 `integration/`
Pruebas de integración entre módulos:
- `test_backend_integration.py` - Integración del backend
- `test_complete_integration.py` - Integración completa del sistema
- Archivos de integración GUI y UI

### 📁 `performance/`
Pruebas de rendimiento y optimización:
- Pruebas de velocidad de algoritmos
- Análisis de uso de memoria
- Benchmarks de comparación

### 📁 `gui/`
Pruebas de interfaz gráfica:
- `test_roi_visualization.py` - Visualización de ROI
- `test_simple_roi_viz.py` - Visualización simplificada
- `test_visualization_features.py` - Características de visualización

### 📁 `data/`
Datos de prueba y fixtures:
- Imágenes de muestra
- Configuraciones de prueba
- Datos de referencia NIST

### 📁 `reports/`
Reportes de ejecución de pruebas:
- Resultados JSON de pruebas
- Reportes de cobertura
- Análisis de rendimiento

### 📁 `legacy/`
Archivos de prueba obsoletos para revisión:
- Pruebas antiguas de integración
- Código de prueba deprecado

## Ejecución de Pruebas

```bash
# Todas las pruebas
pytest

# Pruebas unitarias
pytest tests/unit/

# Pruebas de integración
pytest tests/integration/ -m integration

# Pruebas de rendimiento
pytest tests/performance/ -m performance

# Pruebas GUI (requiere display)
pytest tests/gui/ -m gui

# Pruebas específicas con marcadores
pytest -m "unit and not slow"
pytest -m "integration and nist"
```

## Marcadores Disponibles

- `unit`: Pruebas unitarias rápidas
- `integration`: Pruebas de integración
- `performance`: Pruebas de rendimiento
- `gui`: Pruebas de interfaz gráfica
- `slow`: Pruebas que tardan >30 segundos
- `nist`: Pruebas con estándares NIST

## Respaldo

Los archivos originales se respaldan en `test_backup/` antes de ser movidos.
"""
        
        readme_file = self.tests_dir / "README.md"
        readme_file.write_text(readme_content)
        print("✅ Documentación tests/README.md creada")
        
    def generate_report(self):
        """Genera reporte de la reorganización"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "backup_location": str(self.backup_dir),
            "actions_performed": [
                "Creación de estructura organizada de tests/",
                "Movimiento de archivos de prueba del directorio raíz",
                "Consolidación de integration_legacy en legacy/",
                "Limpieza de directorios vacíos",
                "Creación de pytest.ini",
                "Creación de tests/README.md"
            ],
            "new_structure": {
                "unit": "Pruebas unitarias",
                "integration": "Pruebas de integración", 
                "performance": "Pruebas de rendimiento",
                "gui": "Pruebas de interfaz gráfica",
                "data": "Datos de prueba",
                "reports": "Reportes de pruebas",
                "legacy": "Archivos obsoletos"
            }
        }
        
        report_file = self.root_dir / "test_reorganization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Reporte guardado en: {report_file}")
        
    def run_organization(self):
        """Ejecuta todo el proceso de organización"""
        
        print("🧹 Iniciando organización de archivos de prueba...")
        print("=" * 60)
        
        try:
            self.create_test_structure()
            self.move_root_tests()
            self.consolidate_integration_legacy()
            self.clean_empty_directories()
            self.create_pytest_config()
            self.create_test_readme()
            self.generate_report()
            
            print("=" * 60)
            print("✅ Organización completada exitosamente")
            print(f"📁 Respaldo creado en: {self.backup_dir}")
            
        except Exception as e:
            print(f"❌ Error durante la organización: {e}")
            print(f"📁 Archivos respaldados en: {self.backup_dir}")
            raise

def main():
    """Función principal"""
    organizer = TestOrganizer()
    organizer.run_organization()

if __name__ == "__main__":
    main()