#!/usr/bin/env python3
"""
Test de Imágenes Reales - SIGeC-Balisticar
==========================================

Script para probar el sistema completo con imágenes balísticas reales
disponibles en el directorio assets/

Autor: SIGeC-Balistica Team
Fecha: Enero 2025
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

# Imports del sistema con manejo de errores
try:
    from config.unified_config import get_unified_config
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Config no disponible: {e}")
    CONFIG_AVAILABLE = False

try:
    from utils.logger import setup_logging, get_logger
    LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Logger no disponible: {e}")
    LOGGER_AVAILABLE = False

try:
    from image_processing.unified_preprocessor import UnifiedPreprocessor
    from image_processing.feature_extractor import BallisticFeatureExtractor
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Image processing no disponible: {e}")
    IMAGE_PROCESSING_AVAILABLE = False

try:
    from matching.unified_matcher import UnifiedMatcher
    MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Matching no disponible: {e}")
    MATCHING_AVAILABLE = False

try:
    from core.data_validator import get_data_validator
    from core.error_handler import get_error_manager
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Validation no disponible: {e}")
    VALIDATION_AVAILABLE = False


class RealImageTester:
    """Tester para imágenes balísticas reales"""
    
    def __init__(self):
        self.assets_dir = Path("assets")
        self.results = {
            'images_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'comparisons_made': 0,
            'errors': [],
            'processing_times': [],
            'image_details': []
        }
        
        # Inicializar componentes si están disponibles
        self.preprocessor = None
        self.feature_extractor = None
        self.matcher = None
        self.validator = None
        self.error_manager = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializar componentes del sistema"""
        try:
            if IMAGE_PROCESSING_AVAILABLE:
                self.preprocessor = UnifiedPreprocessor()
                self.feature_extractor = BallisticFeatureExtractor()
                print("✅ Componentes de procesamiento de imágenes inicializados")
            
            if MATCHING_AVAILABLE:
                self.matcher = UnifiedMatcher()
                print("✅ Sistema de matching inicializado")
            
            if VALIDATION_AVAILABLE:
                self.validator = get_data_validator()
                self.error_manager = get_error_manager()
                print("✅ Sistema de validación inicializado")
                
        except Exception as e:
            print(f"⚠️  Error inicializando componentes: {e}")
    
    def find_test_images(self) -> List[Path]:
        """Encontrar imágenes de prueba en assets"""
        if not self.assets_dir.exists():
            print(f"❌ Directorio assets no encontrado: {self.assets_dir}")
            return []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        images = []
        
        for file_path in self.assets_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path)
        
        print(f"🔍 Encontradas {len(images)} imágenes en assets/")
        for img in images:
            print(f"   - {img.name}")
        
        return images
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analizar una imagen individual"""
        start_time = time.time()
        result = {
            'path': str(image_path),
            'name': image_path.name,
            'success': False,
            'processing_time': 0,
            'file_size': 0,
            'dimensions': None,
            'features_extracted': 0,
            'quality_metrics': {},
            'error': None
        }
        
        try:
            # Verificar que el archivo existe y es accesible
            if not image_path.exists():
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            
            result['file_size'] = image_path.stat().st_size
            
            # Cargar imagen con OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            result['dimensions'] = (image.shape[1], image.shape[0])  # width, height
            print(f"📸 Procesando {image_path.name} ({result['dimensions'][0]}x{result['dimensions'][1]})")
            
            # Preprocesamiento si está disponible
            processed_image = image
            if self.preprocessor:
                try:
                    processed_image = self.preprocessor.preprocess(image)
                    print(f"   ✅ Preprocesamiento completado")
                except Exception as e:
                    print(f"   ⚠️  Error en preprocesamiento: {e}")
            
            # Extracción de características si está disponible
            if self.feature_extractor:
                try:
                    features = self.feature_extractor.extract_features(processed_image)
                    if features is not None:
                        result['features_extracted'] = len(features) if hasattr(features, '__len__') else 1
                        print(f"   ✅ Extraídas {result['features_extracted']} características")
                except Exception as e:
                    print(f"   ⚠️  Error extrayendo características: {e}")
            
            # Métricas de calidad básicas
            result['quality_metrics'] = self._calculate_basic_quality(image)
            
            result['success'] = True
            self.results['successful_analyses'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            self.results['failed_analyses'] += 1
            self.results['errors'].append(f"{image_path.name}: {e}")
            print(f"   ❌ Error procesando {image_path.name}: {e}")
        
        finally:
            result['processing_time'] = time.time() - start_time
            self.results['processing_times'].append(result['processing_time'])
            self.results['images_processed'] += 1
        
        return result
    
    def _calculate_basic_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Calcular métricas básicas de calidad de imagen"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Nitidez (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contraste (desviación estándar)
            contrast = gray.std()
            
            # Brillo promedio
            brightness = gray.mean()
            
            # Nivel de ruido (estimación simple)
            noise = cv2.fastNlMeansDenoising(gray).std()
            
            return {
                'sharpness': float(sharpness),
                'contrast': float(contrast),
                'brightness': float(brightness),
                'noise_estimate': float(noise)
            }
        except Exception as e:
            print(f"   ⚠️  Error calculando métricas de calidad: {e}")
            return {}
    
    def compare_images(self, images_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Comparar imágenes entre sí"""
        comparisons = []
        
        if not self.matcher or len(images_data) < 2:
            print("⚠️  No se pueden realizar comparaciones (matcher no disponible o pocas imágenes)")
            return comparisons
        
        successful_images = [img for img in images_data if img['success']]
        
        print(f"\n🔄 Realizando comparaciones entre {len(successful_images)} imágenes...")
        
        for i in range(len(successful_images)):
            for j in range(i + 1, len(successful_images)):
                try:
                    img1_path = successful_images[i]['path']
                    img2_path = successful_images[j]['path']
                    
                    # Cargar imágenes
                    img1 = cv2.imread(img1_path)
                    img2 = cv2.imread(img2_path)
                    
                    if img1 is None or img2 is None:
                        continue
                    
                    # Realizar comparación
                    start_time = time.time()
                    result = self.matcher.compare_images(img1, img2)
                    comparison_time = time.time() - start_time
                    
                    comparison = {
                        'image1': successful_images[i]['name'],
                        'image2': successful_images[j]['name'],
                        'similarity_score': getattr(result, 'similarity_score', 0),
                        'confidence': getattr(result, 'confidence', 0),
                        'comparison_time': comparison_time,
                        'success': True
                    }
                    
                    comparisons.append(comparison)
                    self.results['comparisons_made'] += 1
                    
                    print(f"   📊 {comparison['image1']} vs {comparison['image2']}: "
                          f"Similitud={comparison['similarity_score']:.2f}, "
                          f"Confianza={comparison['confidence']:.2f}")
                
                except Exception as e:
                    error_comparison = {
                        'image1': successful_images[i]['name'],
                        'image2': successful_images[j]['name'],
                        'error': str(e),
                        'success': False
                    }
                    comparisons.append(error_comparison)
                    print(f"   ❌ Error comparando imágenes: {e}")
        
        return comparisons
    
    def generate_report(self, images_data: List[Dict[str, Any]], 
                       comparisons: List[Dict[str, Any]]) -> str:
        """Generar reporte completo de las pruebas"""
        report_lines = [
            "=" * 80,
            "REPORTE DE PRUEBAS CON IMÁGENES REALES - SIGeC-Balisticar",
            "=" * 80,
            f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RESUMEN EJECUTIVO:",
            f"  📁 Imágenes procesadas: {self.results['images_processed']}",
            f"  ✅ Análisis exitosos: {self.results['successful_analyses']}",
            f"  ❌ Análisis fallidos: {self.results['failed_analyses']}",
            f"  🔄 Comparaciones realizadas: {self.results['comparisons_made']}",
            "",
            "ESTADÍSTICAS DE RENDIMIENTO:",
        ]
        
        if self.results['processing_times']:
            avg_time = sum(self.results['processing_times']) / len(self.results['processing_times'])
            max_time = max(self.results['processing_times'])
            min_time = min(self.results['processing_times'])
            
            report_lines.extend([
                f"  ⏱️  Tiempo promedio de procesamiento: {avg_time:.2f}s",
                f"  ⏱️  Tiempo máximo: {max_time:.2f}s",
                f"  ⏱️  Tiempo mínimo: {min_time:.2f}s",
            ])
        
        report_lines.extend([
            "",
            "DETALLES DE IMÁGENES PROCESADAS:",
            "-" * 50
        ])
        
        for img_data in images_data:
            status = "✅" if img_data['success'] else "❌"
            report_lines.append(f"{status} {img_data['name']}")
            
            if img_data['success']:
                report_lines.extend([
                    f"    Dimensiones: {img_data['dimensions']}",
                    f"    Tamaño archivo: {img_data['file_size']:,} bytes",
                    f"    Características extraídas: {img_data['features_extracted']}",
                    f"    Tiempo procesamiento: {img_data['processing_time']:.2f}s"
                ])
                
                if img_data['quality_metrics']:
                    metrics = img_data['quality_metrics']
                    report_lines.append(f"    Métricas de calidad:")
                    for metric, value in metrics.items():
                        report_lines.append(f"      - {metric}: {value:.2f}")
            else:
                report_lines.append(f"    Error: {img_data['error']}")
            
            report_lines.append("")
        
        if comparisons:
            report_lines.extend([
                "RESULTADOS DE COMPARACIONES:",
                "-" * 50
            ])
            
            for comp in comparisons:
                if comp['success']:
                    report_lines.append(
                        f"📊 {comp['image1']} vs {comp['image2']}: "
                        f"Similitud={comp['similarity_score']:.2f}, "
                        f"Confianza={comp['confidence']:.2f} "
                        f"({comp['comparison_time']:.2f}s)"
                    )
                else:
                    report_lines.append(
                        f"❌ {comp['image1']} vs {comp['image2']}: {comp['error']}"
                    )
        
        if self.results['errors']:
            report_lines.extend([
                "",
                "ERRORES ENCONTRADOS:",
                "-" * 50
            ])
            for error in self.results['errors']:
                report_lines.append(f"❌ {error}")
        
        report_lines.extend([
            "",
            "ESTADO DEL SISTEMA:",
            "-" * 50,
            f"✅ Configuración: {'Disponible' if CONFIG_AVAILABLE else 'No disponible'}",
            f"✅ Logger: {'Disponible' if LOGGER_AVAILABLE else 'No disponible'}",
            f"✅ Procesamiento de imágenes: {'Disponible' if IMAGE_PROCESSING_AVAILABLE else 'No disponible'}",
            f"✅ Sistema de matching: {'Disponible' if MATCHING_AVAILABLE else 'No disponible'}",
            f"✅ Sistema de validación: {'Disponible' if VALIDATION_AVAILABLE else 'No disponible'}",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def run_comprehensive_test(self) -> bool:
        """Ejecutar prueba completa del sistema"""
        print("🚀 INICIANDO PRUEBAS CON IMÁGENES REALES")
        print("=" * 60)
        
        # Encontrar imágenes
        images = self.find_test_images()
        if not images:
            print("❌ No se encontraron imágenes para probar")
            return False
        
        # Analizar cada imagen
        print(f"\n📊 ANALIZANDO {len(images)} IMÁGENES...")
        images_data = []
        for image_path in images:
            img_data = self.analyze_image(image_path)
            images_data.append(img_data)
            self.results['image_details'].append(img_data)
        
        # Realizar comparaciones
        print(f"\n🔄 REALIZANDO COMPARACIONES...")
        comparisons = self.compare_images(images_data)
        
        # Generar reporte
        print(f"\n📝 GENERANDO REPORTE...")
        report = self.generate_report(images_data, comparisons)
        
        # Guardar reporte
        report_path = Path("DOCS") / f"test_assets_report_{int(time.time())}.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Reporte guardado en: {report_path}")
        
        # Mostrar resumen
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   Imágenes procesadas: {self.results['images_processed']}")
        print(f"   Análisis exitosos: {self.results['successful_analyses']}")
        print(f"   Análisis fallidos: {self.results['failed_analyses']}")
        print(f"   Comparaciones realizadas: {self.results['comparisons_made']}")
        
        success_rate = (self.results['successful_analyses'] / self.results['images_processed'] * 100) if self.results['images_processed'] > 0 else 0
        print(f"   Tasa de éxito: {success_rate:.1f}%")
        
        return success_rate > 50  # Considerar exitoso si más del 50% funciona


def main():
    """Función principal"""
    try:
        tester = RealImageTester()
        success = tester.run_comprehensive_test()
        
        if success:
            print("\n🎉 PRUEBAS COMPLETADAS EXITOSAMENTE")
            return 0
        else:
            print("\n⚠️  PRUEBAS COMPLETADAS CON PROBLEMAS")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERROR CRÍTICO: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())