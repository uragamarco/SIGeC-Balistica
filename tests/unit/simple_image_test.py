#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Pruebas Simplificado para Análisis de Módulos de Procesamiento de Imágenes
Sistema Balístico Forense SIGeC-Balistica

Este script analiza el estado de los módulos de procesamiento de imágenes
y realiza pruebas básicas con las muestras NIST FADB disponibles.
"""

import os
import sys
import cv2
import numpy as np
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append('/home/marco/SIGeC-Balistica')

class SimpleImageTester:
    """Clase simplificada para probar módulos de procesamiento de imágenes"""
    
    def __init__(self, samples_dir: str = "uploads/Muestras NIST FADB"):
        self.samples_dir = Path(samples_dir)
        self.results_dir = Path("simple_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'modules_analyzed': [],
            'images_tested': [],
            'module_status': {},
            'errors': [],
            'recommendations': []
        }
        
        print(f"Tester inicializado. Directorio de muestras: {self.samples_dir}")
        print(f"Directorio de resultados: {self.results_dir}")
    
    def analyze_module_structure(self, module_path: str) -> Dict[str, Any]:
        """Analiza la estructura de un módulo sin importarlo"""
        module_info = {
            'path': module_path,
            'exists': False,
            'size_bytes': 0,
            'lines_count': 0,
            'classes_found': [],
            'functions_found': [],
            'imports_found': [],
            'potential_issues': []
        }
        
        try:
            if not os.path.exists(module_path):
                module_info['potential_issues'].append("Archivo no encontrado")
                return module_info
            
            module_info['exists'] = True
            module_info['size_bytes'] = os.path.getsize(module_path)
            
            with open(module_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                module_info['lines_count'] = len(lines)
                
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Buscar definiciones de clases
                    if line.startswith('class '):
                        class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                        module_info['classes_found'].append({
                            'name': class_name,
                            'line': i
                        })
                    
                    # Buscar definiciones de funciones
                    elif line.startswith('def '):
                        func_name = line.split('def ')[1].split('(')[0].strip()
                        module_info['functions_found'].append({
                            'name': func_name,
                            'line': i
                        })
                    
                    # Buscar imports
                    elif line.startswith('import ') or line.startswith('from '):
                        module_info['imports_found'].append({
                            'import': line,
                            'line': i
                        })
                    
                    # Buscar posibles problemas
                    if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                        module_info['potential_issues'].append(f"Línea {i}: {line}")
        
        except Exception as e:
            module_info['potential_issues'].append(f"Error leyendo archivo: {str(e)}")
        
        return module_info
    
    def test_basic_imports(self) -> Dict[str, Any]:
        """Prueba las importaciones básicas necesarias"""
        import_results = {
            'opencv': False,
            'numpy': False,
            'scipy': False,
            'sklearn': False,
            'matplotlib': False,
            'pandas': False,
            'pillow': False,
            'errors': []
        }
        
        # Probar OpenCV
        try:
            import cv2
            import_results['opencv'] = True
            print(f"✓ OpenCV versión: {cv2.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"OpenCV: {str(e)}")
            print(f"✗ OpenCV no disponible")
        
        # Probar NumPy
        try:
            import numpy as np
            import_results['numpy'] = True
            print(f"✓ NumPy versión: {np.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"NumPy: {str(e)}")
            print(f"✗ NumPy no disponible")
        
        # Probar SciPy
        try:
            import scipy
            import_results['scipy'] = True
            print(f"✓ SciPy versión: {scipy.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"SciPy: {str(e)}")
            print(f"✗ SciPy no disponible")
        
        # Probar scikit-learn
        try:
            import sklearn
            import_results['sklearn'] = True
            print(f"✓ scikit-learn versión: {sklearn.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"scikit-learn: {str(e)}")
            print(f"✗ scikit-learn no disponible")
        
        # Probar matplotlib
        try:
            import matplotlib
            import_results['matplotlib'] = True
            print(f"✓ matplotlib versión: {matplotlib.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"matplotlib: {str(e)}")
            print(f"✗ matplotlib no disponible")
        
        # Probar pandas
        try:
            import pandas as pd
            import_results['pandas'] = True
            print(f"✓ pandas versión: {pd.__version__}")
        except ImportError as e:
            import_results['errors'].append(f"pandas: {str(e)}")
            print(f"✗ pandas no disponible")
        
        # Probar Pillow
        try:
            from PIL import Image
            import_results['pillow'] = True
            print(f"✓ Pillow disponible")
        except ImportError as e:
            import_results['errors'].append(f"Pillow: {str(e)}")
            print(f"✗ Pillow no disponible")
        
        return import_results
    
    def find_sample_images(self, max_images: int = 5) -> List[Path]:
        """Encuentra imágenes de muestra para las pruebas"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        sample_images = []
        
        try:
            for root, dirs, files in os.walk(self.samples_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_path = Path(root) / file
                        sample_images.append(image_path)
                        
                        if len(sample_images) >= max_images:
                            break
                
                if len(sample_images) >= max_images:
                    break
            
            print(f"Encontradas {len(sample_images)} imágenes de muestra")
            return sample_images[:max_images]
            
        except Exception as e:
            print(f"Error buscando imágenes de muestra: {e}")
            return []
    
    def test_basic_image_processing(self, image_path: Path) -> Dict[str, Any]:
        """Prueba procesamiento básico de imágenes con OpenCV"""
        test_result = {
            'image': str(image_path),
            'success': False,
            'processing_time': 0,
            'image_info': {},
            'basic_features': {},
            'error': None
        }
        
        try:
            start_time = time.time()
            
            # Cargar imagen
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            # Información básica de la imagen
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            test_result['image_info'] = {
                'width': width,
                'height': height,
                'channels': channels,
                'size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Características básicas
            test_result['basic_features'] = {
                'mean_intensity': float(np.mean(gray)),
                'std_intensity': float(np.std(gray)),
                'min_intensity': int(np.min(gray)),
                'max_intensity': int(np.max(gray))
            }
            
            # Detectar bordes básicos
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            test_result['basic_features']['edge_pixels'] = int(edge_pixels)
            test_result['basic_features']['edge_density'] = float(edge_pixels / (width * height))
            
            # Detectar keypoints SIFT básicos
            try:
                sift = cv2.SIFT_create(nfeatures=50)
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                test_result['basic_features']['sift_keypoints'] = len(keypoints)
            except Exception as e:
                test_result['basic_features']['sift_error'] = str(e)
            
            # Detectar keypoints ORB básicos
            try:
                orb = cv2.ORB_create(nfeatures=50)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                test_result['basic_features']['orb_keypoints'] = len(keypoints)
            except Exception as e:
                test_result['basic_features']['orb_error'] = str(e)
            
            processing_time = time.time() - start_time
            test_result.update({
                'success': True,
                'processing_time': processing_time
            })
            
            print(f"✓ Procesamiento básico exitoso para {image_path.name}")
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"✗ Error procesando {image_path.name}: {e}")
        
        return test_result
    
    def analyze_all_modules(self) -> None:
        """Analiza todos los módulos de procesamiento de imágenes"""
        modules_to_analyze = [
            'image_processing/feature_extractor.py',
            'image_processing/ballistic_features.py',
            'image_processing/feature_visualizer.py',
            'image_processing/preprocessing_visualizer.py',
            'image_processing/statistical_analyzer.py',
            'image_processing/statistical_visualizer.py',
            'image_processing/unified_preprocessor.py',
            'image_processing/unified_roi_detector.py'
        ]
        
        print("\nAnalizando estructura de módulos...")
        print("=" * 50)
        
        for module_path in modules_to_analyze:
            module_name = os.path.basename(module_path)
            print(f"\nAnalizando: {module_name}")
            
            module_info = self.analyze_module_structure(module_path)
            self.test_results['modules_analyzed'].append(module_info)
            
            if module_info['exists']:
                print(f"  ✓ Archivo existe ({module_info['size_bytes']} bytes, {module_info['lines_count']} líneas)")
                print(f"  ✓ Clases encontradas: {len(module_info['classes_found'])}")
                print(f"  ✓ Funciones encontradas: {len(module_info['functions_found'])}")
                print(f"  ✓ Imports encontrados: {len(module_info['imports_found'])}")
                
                if module_info['potential_issues']:
                    print(f"  ⚠ Problemas potenciales: {len(module_info['potential_issues'])}")
                    for issue in module_info['potential_issues'][:3]:  # Mostrar solo los primeros 3
                        print(f"    - {issue}")
            else:
                print(f"  ✗ Archivo no encontrado")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Ejecuta análisis comprehensivo del sistema"""
        print("Iniciando análisis comprehensivo del sistema de procesamiento de imágenes")
        print("=" * 70)
        
        # Probar importaciones básicas
        print("\n1. Probando importaciones básicas...")
        import_results = self.test_basic_imports()
        self.test_results['import_status'] = import_results
        
        # Analizar módulos
        print("\n2. Analizando módulos del sistema...")
        self.analyze_all_modules()
        
        # Encontrar y probar imágenes
        print("\n3. Probando procesamiento básico con imágenes reales...")
        sample_images = self.find_sample_images(max_images=3)
        
        if sample_images:
            for i, image_path in enumerate(sample_images, 1):
                print(f"\nProcesando imagen {i}/{len(sample_images)}: {image_path.name}")
                image_result = self.test_basic_image_processing(image_path)
                self.test_results['images_tested'].append(image_result)
        else:
            print("No se encontraron imágenes de muestra para probar")
        
        # Generar recomendaciones
        self.generate_recommendations()
        
        # Guardar resultados
        self.save_results()
        
        return self.test_results
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Verificar importaciones
        import_status = self.test_results.get('import_status', {})
        missing_imports = [lib for lib, status in import_status.items() if not status and lib != 'errors']
        
        if missing_imports:
            recommendations.append(f"Instalar librerías faltantes: {', '.join(missing_imports)}")
        
        # Verificar módulos
        missing_modules = []
        for module in self.test_results['modules_analyzed']:
            if not module['exists']:
                missing_modules.append(os.path.basename(module['path']))
        
        if missing_modules:
            recommendations.append(f"Módulos no encontrados: {', '.join(missing_modules)}")
        
        # Verificar procesamiento de imágenes
        failed_images = [img for img in self.test_results['images_tested'] if not img['success']]
        if failed_images:
            recommendations.append(f"Fallos en procesamiento de {len(failed_images)} imágenes")
        
        # Recomendaciones específicas
        if not import_status.get('opencv', False):
            recommendations.append("CRÍTICO: OpenCV es esencial para el procesamiento de imágenes")
        
        if not import_status.get('numpy', False):
            recommendations.append("CRÍTICO: NumPy es esencial para operaciones numéricas")
        
        # Verificar si hay módulos con muchas líneas (posible refactorización)
        large_modules = [m for m in self.test_results['modules_analyzed'] 
                        if m['exists'] and m['lines_count'] > 1000]
        if large_modules:
            recommendations.append(f"Considerar refactorizar módulos grandes: {[os.path.basename(m['path']) for m in large_modules]}")
        
        self.test_results['recommendations'] = recommendations
    
    def save_results(self):
        """Guarda los resultados del análisis"""
        # Guardar JSON
        results_file = self.results_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Crear reporte en texto
        report_file = self.results_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write("REPORTE DE ANÁLISIS DEL SISTEMA DE PROCESAMIENTO DE IMÁGENES\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"Fecha: {self.test_results['timestamp']}\n\n")
            
            # Estado de importaciones
            f.write("ESTADO DE IMPORTACIONES:\n")
            f.write("-" * 30 + "\n")
            import_status = self.test_results.get('import_status', {})
            for lib, status in import_status.items():
                if lib != 'errors':
                    status_symbol = "✓" if status else "✗"
                    f.write(f"{status_symbol} {lib}: {'Disponible' if status else 'No disponible'}\n")
            
            if import_status.get('errors'):
                f.write("\nErrores de importación:\n")
                for error in import_status['errors']:
                    f.write(f"  - {error}\n")
            
            # Estado de módulos
            f.write(f"\nESTADO DE MÓDULOS ({len(self.test_results['modules_analyzed'])}):\n")
            f.write("-" * 30 + "\n")
            for module in self.test_results['modules_analyzed']:
                module_name = os.path.basename(module['path'])
                if module['exists']:
                    f.write(f"✓ {module_name}: {module['lines_count']} líneas, "
                           f"{len(module['classes_found'])} clases, "
                           f"{len(module['functions_found'])} funciones\n")
                else:
                    f.write(f"✗ {module_name}: No encontrado\n")
            
            # Resultados de imágenes
            if self.test_results['images_tested']:
                f.write(f"\nPROCESAMIENTO DE IMÁGENES ({len(self.test_results['images_tested'])}):\n")
                f.write("-" * 30 + "\n")
                for img_result in self.test_results['images_tested']:
                    img_name = os.path.basename(img_result['image'])
                    if img_result['success']:
                        f.write(f"✓ {img_name}: {img_result['processing_time']:.3f}s\n")
                        if 'basic_features' in img_result:
                            features = img_result['basic_features']
                            f.write(f"  - Dimensiones: {img_result['image_info']['width']}x{img_result['image_info']['height']}\n")
                            f.write(f"  - SIFT keypoints: {features.get('sift_keypoints', 'N/A')}\n")
                            f.write(f"  - ORB keypoints: {features.get('orb_keypoints', 'N/A')}\n")
                    else:
                        f.write(f"✗ {img_name}: {img_result.get('error', 'Error desconocido')}\n")
            
            # Recomendaciones
            if self.test_results['recommendations']:
                f.write(f"\nRECOMENDACIONES:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(self.test_results['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
        
        print(f"\nResultados guardados en: {results_file}")
        print(f"Reporte guardado en: {report_file}")

def main():
    """Función principal"""
    print("Iniciando análisis simplificado del sistema de procesamiento de imágenes...")
    print("=" * 70)
    
    # Crear tester
    tester = SimpleImageTester()
    
    # Ejecutar análisis
    results = tester.run_comprehensive_analysis()
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("RESUMEN DEL ANÁLISIS:")
    print("-" * 30)
    
    # Estado de importaciones
    import_status = results.get('import_status', {})
    available_libs = sum(1 for lib, status in import_status.items() if status and lib != 'errors')
    total_libs = len([lib for lib in import_status.keys() if lib != 'errors'])
    print(f"Librerías disponibles: {available_libs}/{total_libs}")
    
    # Estado de módulos
    existing_modules = sum(1 for m in results['modules_analyzed'] if m['exists'])
    total_modules = len(results['modules_analyzed'])
    print(f"Módulos encontrados: {existing_modules}/{total_modules}")
    
    # Procesamiento de imágenes
    if results['images_tested']:
        successful_images = sum(1 for img in results['images_tested'] if img['success'])
        total_images = len(results['images_tested'])
        print(f"Imágenes procesadas exitosamente: {successful_images}/{total_images}")
    
    # Recomendaciones
    if results['recommendations']:
        print(f"Recomendaciones generadas: {len(results['recommendations'])}")
        print("\nPrincipales recomendaciones:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nResultados detallados en: {tester.results_dir}")

if __name__ == "__main__":
    main()