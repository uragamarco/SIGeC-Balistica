#!/usr/bin/env python3
"""
Plan de Reorganización de Tests - SIGeC-Balisticar
==================================================

Este script identifica y reorganiza los tests duplicados y desorganizados
del sistema para crear una estructura más limpia y mantenible.

Autor: SIGeC-BalisticaTeam
Versión: 1.0.0
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set
import re

class TestReorganizer:
    """Reorganizador de estructura de tests"""
    
    def __init__(self, tests_dir: str = "tests"):
        self.tests_dir = Path(tests_dir)
        self.duplicates = {}
        self.consolidation_plan = {}
        
    def analyze_test_structure(self) -> Dict[str, List[str]]:
        """Analiza la estructura actual de tests e identifica duplicados"""
        
        # Mapeo de archivos duplicados identificados
        duplicates_map = {
            # Tests de integración backend duplicados
            "backend_integration": [
                "tests/integration/test_backend_integration.py",
                "tests/integration/test_backend_integration_consolidated.py", 
                "tests/legacy/test_backend_integration.py"
            ],
            
            # Tests de GUI duplicados
            "gui_tests": [
                "tests/integration/test_gui_comprehensive.py",
                "tests/integration/test_ui_comprehensive.py",
                "tests/integration/test_gui_headless.py",
                "tests/integration/test_ui_headless.py"
            ],
            
            # Tests de integración general duplicados
            "integration_tests": [
                "tests/integration/test_integration.py",
                "tests/integration/test_complete_integration.py",
                "tests/integration/test_integration_headless.py",
                "tests/test_complete_integration_consolidated.py"
            ],
            
            # Tests de frontend duplicados
            "frontend_tests": [
                "tests/integration/test_frontend_integration_consolidated.py",
                "tests/integration/test_ui_integration.py"
            ],
            
            # Tests de performance duplicados
            "performance_tests": [
                "tests/integration/test_performance_integration_consolidated.py",
                "tests/test_performance_benchmarks.py"
            ],
            
            # Tests de imágenes NIST duplicados
            "nist_tests": [
                "tests/test_nist_real_images.py",
                "tests/test_nist_real_images_simple.py",
                "tests/test_nist_standards.py",
                "tests/test_nist_validation.py",
                "tests/test_nist_statistical_integration.py"
            ],
            
            # Tests de procesamiento de imágenes duplicados
            "image_processing_tests": [
                "tests/test_image_processing.py",
                "tests/unit/test_image_processing.py"
            ]
        }
        
        return duplicates_map
    
    def create_consolidation_plan(self) -> Dict[str, Dict]:
        """Crea un plan de consolidación para eliminar duplicados"""
        
        plan = {
            # Consolidar tests de backend
            "backend_integration": {
                "target": "tests/integration/test_backend_consolidated.py",
                "sources": [
                    "tests/integration/test_backend_integration.py",
                    "tests/integration/test_backend_integration_consolidated.py"
                ],
                "legacy": ["tests/legacy/test_backend_integration.py"],
                "description": "Consolidar todos los tests de integración backend"
            },
            
            # Consolidar tests de GUI
            "gui_integration": {
                "target": "tests/integration/test_gui_consolidated.py", 
                "sources": [
                    "tests/integration/test_gui_comprehensive.py",
                    "tests/integration/test_ui_comprehensive.py"
                ],
                "headless": [
                    "tests/integration/test_gui_headless.py",
                    "tests/integration/test_ui_headless.py"
                ],
                "description": "Consolidar tests de GUI en modo normal y headless"
            },
            
            # Consolidar tests de integración completa
            "complete_integration": {
                "target": "tests/integration/test_complete_integration.py",
                "sources": [
                    "tests/integration/test_integration.py",
                    "tests/test_complete_integration_consolidated.py"
                ],
                "headless": ["tests/integration/test_integration_headless.py"],
                "description": "Tests de integración completa del sistema"
            },
            
            # Consolidar tests de performance
            "performance": {
                "target": "tests/performance/test_performance_suite.py",
                "sources": [
                    "tests/integration/test_performance_integration_consolidated.py",
                    "tests/test_performance_benchmarks.py"
                ],
                "description": "Suite completa de tests de performance"
            },
            
            # Consolidar tests NIST
            "nist_validation": {
                "target": "tests/validation/test_nist_compliance.py",
                "sources": [
                    "tests/test_nist_standards.py",
                    "tests/test_nist_validation.py"
                ],
                "real_images": [
                    "tests/test_nist_real_images.py",
                    "tests/test_nist_real_images_simple.py"
                ],
                "statistical": ["tests/test_nist_statistical_integration.py"],
                "description": "Tests de cumplimiento y validación NIST"
            }
        }
        
        return plan
    
    def create_new_structure(self) -> Dict[str, List[str]]:
        """Define la nueva estructura organizacional de tests"""
        
        structure = {
            "tests/unit/": [
                "test_image_processing.py",
                "test_feature_extraction.py", 
                "test_matching_algorithms.py",
                "test_database_operations.py",
                "test_configuration.py"
            ],
            
            "tests/integration/": [
                "test_backend_consolidated.py",
                "test_gui_consolidated.py", 
                "test_complete_integration.py",
                "test_pipeline_integration.py"
            ],
            
            "tests/performance/": [
                "test_performance_suite.py",
                "test_memory_optimization.py",
                "test_benchmark_system.py"
            ],
            
            "tests/validation/": [
                "test_nist_compliance.py",
                "test_scientific_validation.py",
                "test_statistical_analysis.py"
            ],
            
            "tests/gui/": [
                "test_roi_visualization.py",
                "test_visualization_features.py",
                "test_interactive_components.py"
            ],
            
            "tests/headless/": [
                "test_headless_gui.py",
                "test_headless_integration.py"
            ]
        }
        
        return structure
    
    def generate_reorganization_script(self) -> str:
        """Genera script de reorganización"""
        
        script = '''#!/usr/bin/env python3
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
        print(f"\\n{category}:")
        for file_path in files:
            if Path(file_path).exists():
                print(f"  - {file_path}")
    
    print("\\n✅ Reorganización completada. Revisar consolidaciones manuales.")

if __name__ == "__main__":
    reorganize_tests()
'''
        
        return script
    
    def get_cleanup_recommendations(self) -> List[str]:
        """Genera recomendaciones de limpieza"""
        
        recommendations = [
            "🔄 REORGANIZACIÓN DE TESTS - PLAN DE ACCIÓN",
            "=" * 50,
            "",
            "1. ARCHIVOS DUPLICADOS IDENTIFICADOS:",
            "   - 3 versiones de test_backend_integration.py",
            "   - 4 versiones de tests GUI (comprehensive, ui, headless)",
            "   - 5 versiones de tests NIST",
            "   - 2 versiones de test_image_processing.py",
            "",
            "2. ACCIONES RECOMENDADAS:",
            "   ✅ Consolidar tests de backend en test_backend_consolidated.py",
            "   ✅ Unificar tests GUI en test_gui_consolidated.py",
            "   ✅ Crear suite NIST en test_nist_compliance.py",
            "   ✅ Mover archivos legacy a tests/legacy_backup/",
            "",
            "3. NUEVA ESTRUCTURA PROPUESTA:",
            "   tests/",
            "   ├── unit/           # Tests unitarios específicos",
            "   ├── integration/    # Tests de integración consolidados", 
            "   ├── performance/    # Tests de rendimiento",
            "   ├── validation/     # Tests de validación científica",
            "   ├── gui/           # Tests específicos de GUI",
            "   ├── headless/      # Tests en modo headless",
            "   └── legacy_backup/ # Archivos legacy respaldados",
            "",
            "4. BENEFICIOS ESPERADOS:",
            "   - Reducción de ~40% en archivos de test duplicados",
            "   - Estructura más clara y mantenible",
            "   - Eliminación de redundancias",
            "   - Mejor organización por categorías funcionales"
        ]
        
        return recommendations

def main():
    """Función principal"""
    reorganizer = TestReorganizer()
    
    # Analizar estructura actual
    duplicates = reorganizer.analyze_test_structure()
    
    # Crear plan de consolidación
    plan = reorganizer.create_consolidation_plan()
    
    # Generar recomendaciones
    recommendations = reorganizer.get_cleanup_recommendations()
    
    # Mostrar resultados
    print("\n".join(recommendations))
    
    # Generar script de reorganización
    script = reorganizer.generate_reorganization_script()
    
    with open("tests/reorganize_tests_auto.py", "w", encoding="utf-8") as f:
        f.write(script)
    
    print(f"\n📝 Script de reorganización generado: tests/reorganize_tests_auto.py")
    print("💡 Ejecutar el script para aplicar los cambios automáticos")

if __name__ == "__main__":
    main()