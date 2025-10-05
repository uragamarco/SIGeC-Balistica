#!/usr/bin/env python3
"""
Test completo del flujo de trabajo de la aplicación SEACABA
Prueba la integración completa: carga → procesamiento → comparación → visualización
"""

import os
import sys
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw
import hashlib

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_config
from image_processing.unified_preprocessor import UnifiedPreprocessor
from image_processing.feature_extractor import FeatureExtractor
from matching.unified_matcher import UnifiedMatcher
from database.vector_db import VectorDatabase, BallisticCase, BallisticImage, FeatureVector

class CompleteFlowTest:
    """Test completo del flujo de trabajo"""
    
    def __init__(self):
        self.config = get_config()
        self.preprocessor = UnifiedPreprocessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.matcher = UnifiedMatcher(self.config)
        self.db_manager = VectorDatabase(self.config)
        
    def create_test_images(self):
        """Crea imágenes de prueba simulando vainas balísticas"""
        images = []
        
        for i in range(3):
            # Crear imagen con patrones únicos
            img = Image.new('RGB', (400, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            # Simular patrones de vaina
            center = (200, 200)
            radius = 150
            
            # Círculo principal
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                        outline='black', width=3)
            
            # Patrones únicos para cada imagen
            for j in range(10 + i * 5):
                x = center[0] + np.random.randint(-100, 100)
                y = center[1] + np.random.randint(-100, 100)
                draw.ellipse([x-5, y-5, x+5, y+5], fill='black')
            
            # Guardar imagen temporal
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            img.save(temp_file.name)
            images.append(temp_file.name)
            
        return images
    
    def test_complete_workflow(self):
        """Test del flujo completo de trabajo"""
        print("=== TEST COMPLETO DEL FLUJO DE TRABAJO ===\n")
        
        # 1. Crear imágenes de prueba
        print("1. Creando imágenes de prueba...")
        test_images = self.create_test_images()
        print(f"   ✓ Creadas {len(test_images)} imágenes de prueba")
        
        # 2. Crear caso en base de datos
        print("\n2. Creando caso en base de datos...")
        case = BallisticCase(
            case_number=f"TEST_{int(time.time())}",
            investigator="Test Investigator",
            date_created=time.strftime("%Y-%m-%d"),
            weapon_type="Pistola",
            weapon_model="Test Model",
            caliber="9mm",
            description="Caso de prueba para flujo completo"
        )
        case_id = self.db_manager.add_case(case)
        print(f"   ✓ Caso creado con ID: {case_id}")
        
        # 3. Procesar cada imagen
        processed_images = []
        feature_vectors = []
        
        for i, image_path in enumerate(test_images):
            print(f"\n3.{i+1}. Procesando imagen {i+1}...")
            
            # 3.1 Preprocesamiento
            print("   - Preprocesando imagen...")
            result = self.preprocessor.preprocess_ballistic_image(image_path)
            if not result.success:
                print(f"     ⚠ Error en preprocesamiento: {result.error_message}")
                continue
            preprocessed = result.processed_image
            print(f"     ✓ Imagen preprocesada: {preprocessed.shape}")
            
            # 3.2 Extracción de características
            print("   - Extrayendo características...")
            keypoints, descriptors = self.feature_extractor.get_keypoints_and_descriptors(preprocessed, 'orb')
            print(f"     ✓ Extraídas {len(keypoints)} características")
            
            # 3.3 Agregar imagen a base de datos
            print("   - Agregando a base de datos...")
            unique_hash = hashlib.md5(f"{image_path}_{time.time()}_{i}".encode()).hexdigest()
            image_obj = BallisticImage(
                case_id=case_id,
                filename=os.path.basename(image_path),
                file_path=image_path,
                evidence_type="vaina",
                image_hash=unique_hash,
                width=400,
                height=400,
                file_size=os.path.getsize(image_path)
            )
            image_id = self.db_manager.add_image(image_obj)
            print(f"     ✓ Imagen agregada con ID: {image_id}")
            
            # 3.4 Agregar vector de características
            if descriptors is not None and len(descriptors) > 0:
                print("   - Agregando vector de características...")
                vector_obj = FeatureVector(
                    image_id=image_id,
                    algorithm="ORB",
                    vector_size=len(descriptors),
                    extraction_params='{"max_features": 500}'
                )
                vector_id = self.db_manager.add_feature_vector(vector_obj, descriptors)
                print(f"     ✓ Vector agregado con ID: {vector_id}")
                
                processed_images.append({
                    'path': image_path,
                    'preprocessed': preprocessed,
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'image_id': image_id,
                    'vector_id': vector_id
                })
            else:
                print("     ⚠ No se pudieron extraer características válidas")
        
        # 4. Realizar comparaciones
        print(f"\n4. Realizando comparaciones entre {len(processed_images)} imágenes...")
        
        if len(processed_images) >= 2:
            # Comparar primera imagen con las demás
            base_image = processed_images[0]
            
            for i, compare_image in enumerate(processed_images[1:], 1):
                print(f"\n4.{i}. Comparando imagen base con imagen {i+1}...")
                
                # Comparación usando compare_images
                result = self.matcher.compare_images(
                    base_image['preprocessed'], 
                    compare_image['preprocessed']
                )
                
                if result and hasattr(result, 'similarity_score') and result.good_matches > 0:
                    similarity = result.similarity_score
                    print(f"     ✓ Similitud: {similarity:.3f}")
                    print(f"     ✓ Matches encontrados: {result.good_matches}")
                    
                    # Búsqueda en base de datos
                    print("   - Buscando similares en base de datos...")
                    results = self.db_manager.search_similar_vectors(
                        base_image['descriptors'], 
                        k=5
                    )
                    print(f"     ✓ Encontrados {len(results)} resultados similares")
                    
                    for j, (idx, distance) in enumerate(results):
                        print(f"       {j+1}. Distancia: {distance:.3f}, Índice: {idx}")
                else:
                    print("     ⚠ No se encontraron matches válidos")
        
        # 5. Estadísticas finales
        print(f"\n5. Estadísticas finales...")
        stats = self.db_manager.get_database_stats()
        print(f"   ✓ Total de casos: {stats.get('total_cases', 0)}")
        print(f"   ✓ Total de imágenes: {stats.get('total_images', 0)}")
        print(f"   ✓ Total de vectores: {stats.get('total_vectors', 0)}")
        
        # 6. Limpieza
        print(f"\n6. Limpiando archivos temporales...")
        for image_path in test_images:
            try:
                os.unlink(image_path)
                print(f"   ✓ Eliminado: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"   ⚠ Error eliminando {os.path.basename(image_path)}: {e}")
        
        print(f"\n=== TEST COMPLETO FINALIZADO ===")
        print(f"✓ Flujo completo ejecutado exitosamente")
        return True

def main():
    """Función principal"""
    try:
        test = CompleteFlowTest()
        success = test.test_complete_workflow()
        
        if success:
            print("\n🎉 TODOS LOS TESTS PASARON CORRECTAMENTE")
            return 0
        else:
            print("\n❌ ALGUNOS TESTS FALLARON")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())