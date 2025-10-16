#!/usr/bin/env python3
"""
Script de corrección rápida para errores críticos en SIGeC-Balística
Ejecutar: python fix_critical_errors.py
"""

import os
import sys
from pathlib import Path

def fix_comparison_tab():
    """Corrige la inicialización faltante en ComparisonTab"""
    file_path = Path("gui/comparison_tab.py")
    
    if not file_path.exists():
        print(f"❌ No se encontró {file_path}")
        return False
    
    print(f"🔧 Corrigiendo {file_path}...")
    
    # Leer contenido actual
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el método __init__ y agregar inicializaciones faltantes
    init_method = "def __init__(self):"
    setup_ui_call = "self.setup_ui()"
    
    if init_method in content and setup_ui_call in content:
        # Agregar inicializaciones antes de setup_ui()
        additions = """
        # Inicializar widgets faltantes para evitar AttributeError
        self.cmc_visualization = CMCVisualizationWidget()
        
        # Crear query_image_viewer si no existe
        try:
            from .image_viewer import ImageViewer
            self.query_image_viewer = ImageViewer()
        except ImportError:
            # Fallback a QLabel si ImageViewer no está disponible
            from PyQt5.QtWidgets import QLabel
            self.query_image_viewer = QLabel("Image Viewer")
        
        """
        
        content = content.replace(
            setup_ui_call,
            additions + "        " + setup_ui_call
        )
        
        # Escribir archivo corregido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {file_path} corregido")
        return True
    else:
        print(f"❌ No se pudo localizar el método __init__ en {file_path}")
        return False

def fix_vector_db():
    """Corrige el error de get_absolute_path en VectorDatabase"""
    file_path = Path("database/vector_db.py")
    
    if not file_path.exists():
        print(f"❌ No se encontró {file_path}")
        return False
    
    print(f"🔧 Corrigiendo {file_path}...")
    
    # Leer contenido actual
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar las líneas problemáticas
    old_code = """        # Configurar rutas de base de datos
        self.db_path = self.config.get_absolute_path(self.config.database.sqlite_path)
        self.faiss_path = self.config.get_absolute_path(self.config.database.faiss_index_path)"""
    
    new_code = """        # Configurar rutas de base de datos
        if hasattr(self.config, 'get_absolute_path'):
            # Es UnifiedConfig
            self.db_path = self.config.get_absolute_path(self.config.database.sqlite_path)
            self.faiss_path = self.config.get_absolute_path(self.config.database.faiss_index_path)
        else:
            # Es DatabaseConfig directamente - usar rutas como están
            from pathlib import Path
            self.db_path = Path(self.config.sqlite_path) if hasattr(self.config, 'sqlite_path') else Path(self.config.database.sqlite_path)
            self.faiss_path = Path(self.config.faiss_index_path) if hasattr(self.config, 'faiss_index_path') else Path(self.config.database.faiss_index_path)"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Escribir archivo corregido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {file_path} corregido")
        return True
    else:
        print(f"❌ No se pudo localizar el código problemático en {file_path}")
        return False

def fix_yaml_config():
    """Corrige la configuración YAML problemática"""
    file_path = Path("config/unified_config.yaml")
    
    if not file_path.exists():
        print(f"❌ No se encontró {file_path}")
        return False
    
    print(f"🔧 Corrigiendo {file_path}...")
    
    # Leer contenido actual
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Reemplazar la línea problemática
    old_line = "  algorithm: !!python/object/apply:matching.unified_matcher.AlgorithmType"
    new_line = "  algorithm: \"ORB\""
    
    if old_line in content:
        # Reemplazar la línea y la siguiente que contiene "  - ORB"
        lines = content.split('\n')
        new_lines = []
        skip_next = False
        
        for line in lines:
            if skip_next:
                skip_next = False
                continue
            
            if old_line.strip() in line:
                new_lines.append(line.replace(old_line.strip(), new_line.strip()))
                skip_next = True  # Saltar la línea "  - ORB"
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # Escribir archivo corregido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {file_path} corregido")
        return True
    else:
        print(f"❌ No se pudo localizar la línea problemática en {file_path}")
        return False

def main():
    """Función principal del script de corrección"""
    print("🚀 Iniciando corrección de errores críticos en SIGeC-Balística")
    print("=" * 60)
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    corrections = [
        ("ComparisonTab AttributeError", fix_comparison_tab),
        ("VectorDatabase get_absolute_path", fix_vector_db),
        ("YAML AlgorithmType", fix_yaml_config)
    ]
    
    success_count = 0
    
    for name, fix_func in corrections:
        print(f"\n📋 Corrigiendo: {name}")
        try:
            if fix_func():
                success_count += 1
            else:
                print(f"⚠️  Corrección de {name} falló")
        except Exception as e:
            print(f"❌ Error al corregir {name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Resumen: {success_count}/{len(corrections)} correcciones aplicadas")
    
    if success_count == len(corrections):
        print("🎉 ¡Todas las correcciones críticas aplicadas exitosamente!")
        print("\n📝 Próximos pasos recomendados:")
        print("   1. Ejecutar: python -m gui.reports_tab")
        print("   2. Verificar que no hay más AttributeErrors")
        print("   3. Instalar folium: pip install folium")
    else:
        print("⚠️  Algunas correcciones fallaron. Revisar manualmente.")
    
    return success_count == len(corrections)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)