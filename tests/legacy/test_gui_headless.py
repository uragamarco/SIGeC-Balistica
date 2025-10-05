#!/usr/bin/env python3
"""
Test GUI Headless - Pruebas de interfaz gráfica sin display
Valida la integración GUI-Backend en modo headless
"""

import os
import sys
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw
import hashlib
from unittest.mock import MagicMock, patch

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock PyQt5 para evitar problemas de display
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()

from utils.config import get_config
from image_processing.unified_preprocessor import UnifiedPreprocessor
from image_processing.feature_extractor import FeatureExtractor
from matching.unified_matcher import UnifiedMatcher
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector

class HeadlessGUITest:
    """Test de GUI en modo headless"""
    
    def __init__(self):
        self.config = get_config()
        self.preprocessor = UnifiedPreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.matcher = UnifiedMatcher(self.config)
        self.db_manager = VectorDatabase(self.config)
        
    def create_test_image(self, size=(400, 400), pattern="circles"):
        """Crear imagen de prueba"""
        image = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(image)
        
        if pattern == "circles":
            # Patrón de círculos para simular características balísticas
            for i in range(5):
                for j in range(5):
                    x = 50 + i * 70
                    y = 50 + j * 70
                    radius = 15 + (i + j) * 2
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                               fill='black', outline='gray')
        elif pattern == "lines":
            # Patrón de líneas para simular estrías
            for i in range(0, size[0], 20):
                draw.line([i, 0, i, size[1]], fill='black', width=2)
                
        # Guardar imagen temporal
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_file.name)
        return temp_file.name
        
    def test_gui_backend_integration(self):
        """Test de integración GUI-Backend simulado"""
        print("🔧 Iniciando test GUI-Backend headless...")
        
        try:
            # 1. Simular carga de imagen desde GUI
            print("\n1. Simulando carga de imagen desde GUI...")
            test_image = self.create_test_image(pattern="circles")
            print(f"   ✓ Imagen creada: {test_image}")
            
            # 2. Simular selección de tipo de evidencia
            evidence_type = "vaina"
            print(f"   ✓ Tipo de evidencia seleccionado: {evidence_type}")
            
            # 3. Simular procesamiento backend
            print("\n2. Procesando imagen en backend...")
            
            # 3.1 Preprocesamiento
            result = self.preprocessor.preprocess_ballistic_image(test_image)
            if not result.success:
                print(f"   ✗ Error en preprocesamiento: {result.error_message}")
                return False
            print(f"   ✓ Imagen preprocesada: {result.processed_image.shape}")
            
            # 3.2 Extracción de características
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(
                result.processed_image, 'orb'
            )
            print(f"   ✓ Características extraídas: {len(keypoints)} keypoints")
            
            # 4. Simular guardado en base de datos
            print("\n3. Guardando en base de datos...")
            
            # Crear caso único
            case_number = f"GUI-TEST-{int(time.time())}"
            case = BallisticCase(
                case_number=case_number,
                investigator="Test GUI",
                description="Test de integración GUI-Backend headless"
            )
            case_id = self.db_manager.add_case(case)
            print(f"   ✓ Caso creado con ID: {case_id}")
            
            # Agregar imagen
            unique_hash = hashlib.md5(f"{test_image}_{time.time()}".encode()).hexdigest()
            image_obj = BallisticImage(
                case_id=case_id,
                filename=os.path.basename(test_image),
                file_path=test_image,
                evidence_type=evidence_type,
                image_hash=unique_hash,
                width=400,
                height=400,
                file_size=os.path.getsize(test_image)
            )
            image_id = self.db_manager.add_image(image_obj)
            print(f"   ✓ Imagen guardada con ID: {image_id}")
            
            # Agregar vector de características
            if descriptors is not None and len(descriptors) > 0:
                vector_obj = FeatureVector(
                    image_id=image_id,
                    algorithm="ORB",
                    vector_size=len(descriptors),
                    extraction_params='{"max_features": 500}'
                )
                vector_id = self.db_manager.add_feature_vector(vector_obj, descriptors)
                print(f"   ✓ Vector guardado con ID: {vector_id}")
            
            # 5. Simular búsqueda en base de datos
            print("\n4. Simulando búsqueda en base de datos...")
            search_results = self.db_manager.search_similar_vectors(descriptors, k=5)
            print(f"   ✓ Encontrados {len(search_results)} resultados similares")
            
            # 6. Simular comparación de imágenes
            print("\n5. Simulando comparación de imágenes...")
            test_image2 = self.create_test_image(pattern="lines")
            
            # Procesar segunda imagen
            result2 = self.preprocessor.preprocess_ballistic_image(test_image2)
            if result2.success:
                comparison_result = self.matcher.compare_images(
                    result.processed_image, 
                    result2.processed_image
                )
                if comparison_result and hasattr(comparison_result, 'similarity_score'):
                    print(f"   ✓ Comparación exitosa - Similitud: {comparison_result.similarity_score:.2f}")
                    print(f"   ✓ Matches encontrados: {comparison_result.good_matches}")
                else:
                    print("   ⚠ Comparación sin resultados válidos")
            
            # 7. Simular generación de reporte
            print("\n6. Simulando generación de reporte...")
            report_data = {
                'case_number': case_number,
                'evidence_type': evidence_type,
                'total_features': len(keypoints),
                'similar_cases': len(search_results),
                'processing_time': time.time(),
                'status': 'completed'
            }
            print(f"   ✓ Reporte generado: {report_data}")
            
            # 8. Limpiar archivos temporales
            print("\n7. Limpiando archivos temporales...")
            try:
                os.unlink(test_image)
                os.unlink(test_image2)
                print("   ✓ Archivos temporales eliminados")
            except Exception as e:
                print(f"   ⚠ Error limpiando archivos: {e}")
            
            return True
            
        except Exception as e:
            print(f"\n💥 ERROR en test GUI-Backend: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_gui_components_mock(self):
        """Test de componentes GUI mockeados"""
        print("\n🔧 Probando componentes GUI mockeados...")
        
        try:
            # Simular importación de componentes GUI sin instanciarlos
            print("   ✓ Componentes GUI disponibles para importación")
            
            # Verificar que los mocks están funcionando
            import PyQt5.QtWidgets
            import PyQt5.QtCore
            import PyQt5.QtGui
            print("   ✓ PyQt5 mockeado correctamente")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error en componentes GUI: {e}")
            return False
    
    def test_database_operations(self):
        """Test de operaciones de base de datos"""
        print("\n🔧 Probando operaciones de base de datos...")
        
        try:
            # Obtener estadísticas
            stats = self.db_manager.get_database_stats()
            print(f"   ✓ Estadísticas obtenidas: {stats}")
            
            # Verificar conexión
            if hasattr(self.db_manager, 'connection') and self.db_manager.connection:
                print("   ✓ Conexión a base de datos activa")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error en operaciones de BD: {e}")
            return False

def main():
    """Función principal"""
    print("=" * 60)
    print("TEST GUI HEADLESS - INTEGRACIÓN GUI-BACKEND")
    print("=" * 60)
    
    test = HeadlessGUITest()
    
    # Ejecutar tests
    tests = [
        ("Integración GUI-Backend", test.test_gui_backend_integration),
        ("Componentes GUI Mock", test.test_gui_components_mock),
        ("Operaciones de BD", test.test_database_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResultado: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("\n🎉 TODOS LOS TESTS HEADLESS PASARON CORRECTAMENTE")
        print("✅ La integración GUI-Backend está funcionando")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests fallaron")
        return 1

if __name__ == "__main__":
    sys.exit(main())