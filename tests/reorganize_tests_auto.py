#!/usr/bin/env python3
"""
Script de Reorganización Automática de Tests
============================================
"""

import os
import shutil
from pathlib import Path

def reorganize_tests():
    """Ejecuta la reorganización de tests"""
    
    # 1. Crear nueva estructura de directorios
    directories = [
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/validation",
        "tests/gui",
        "tests/headless",
        "tests/legacy_backup"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    # 2. Mover archivos legacy a backup
    legacy_files = [
        "tests/legacy/test_backend_integration.py",
        "tests/legacy/test_final_integration.py"
    ]
    
    for file_path in legacy_files:
        if Path(file_path).exists():
            shutil.move(file_path, f"tests/legacy_backup/{Path(file_path).name}")
    
    # 3. Consolidar archivos duplicados (manual)
    print("⚠️  Los siguientes archivos requieren consolidación manual:")
    
    duplicates = {
        "Backend Integration": [
            "tests/integration/test_backend_integration.py",
            "tests/integration/test_backend_integration_consolidated.py"
        ],
        "GUI Tests": [
            "tests/integration/test_gui_comprehensive.py", 
            "tests/integration/test_ui_comprehensive.py"
        ],
        "NIST Tests": [
            "tests/test_nist_real_images.py",
            "tests/test_nist_real_images_simple.py"
        ]
    }
    
    for category, files in duplicates.items():
        print(f"\n{category}:")
        for file_path in files:
            if Path(file_path).exists():
                print(f"  - {file_path}")
    
    print("\n✅ Reorganización completada. Revisar consolidaciones manuales.")

if __name__ == "__main__":
    reorganize_tests()
