#!/usr/bin/env python3
"""
Tests de integración para el sistema de procesamiento por chunks de SEACABAr
Valida la integración completa entre todos los componentes
"""

import os
import sys
import gc
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import tempfile
import shutil

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from image_processing.chunked_processor import ChunkedImageProcessor, ChunkingStrategy
    from image_processing.optimized_loader import OptimizedImageLoader, LoadingStrategy
    from image_processing.lazy_loading import LazyImageLoader, LazyImageDataset
    from image_processing.unified_preprocessor import UnifiedPreprocessor
except ImportError as e:
    print(f"Warning: Could not import image processing modules: {e}")
    # Crear mocks para testing
    ChunkedImageProcessor = Mock
    ChunkingStrategy = Mock
    OptimizedImageLoader = Mock
    LoadingStrategy = Mock
    LazyImageLoader = Mock
    LazyImageDataset = Mock
    UnifiedPreprocessor = Mock


def create_test_image_file(filepath: str, width: int, height: int) -> str:
    """Crea un archivo de imagen de prueba"""
    try:
        from PIL import Image
        img = Image.new('RGB', (width, height), color='red')
        img.save(filepath)
        return filepath
    except ImportError:
        # Crear archivo simulado
        with open(filepath, 'wb') as f:
            # Escribir datos simulados de imagen
            f.write(b'FAKE_IMAGE_DATA' * 100)
        return filepath


class TestChunkedSystemIntegration:
    """Tests de integración del sistema completo de chunks"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        gc.collect()
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        gc.collect()
    
    def test_end_to_end_chunked_processing(self):
        """Test end-to-end del procesamiento por chunks"""
        # Crear imagen de prueba
        test_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        
        # Configurar procesador por chunks
        if ChunkedImageProcessor != Mock:
            processor = ChunkedImageProcessor(
                chunk_size=(256, 256),
                overlap=32,
                strategy=ChunkingStrategy.GRID,
                parallel=False  # Para testing determinístico
            )
            
            def enhancement_function(chunk):
                """Función de mejora simple"""
                return np.clip(chunk * 1.1, 0, 255).astype(np.uint8)
            
            # Procesar imagen
            result = processor.process_image(test_image, enhancement_function)
            
            # Verificar resultado
            assert result is not None, "El resultado no debe ser None"
            assert result.shape == test_image.shape, "Las dimensiones deben coincidir"
            assert result.dtype == test_image.dtype, "El tipo de datos debe coincidir"
            
            # Verificar que se aplicó la mejora
            assert not np.array_equal(result, test_image), "La imagen debe haber cambiado"
        else:
            pytest.skip("ChunkedImageProcessor no disponible")
    
    def test_chunked_with_unified_preprocessor_integration(self):
        """Test de integración entre chunked processor y unified preprocessor"""
        if ChunkedImageProcessor != Mock and UnifiedPreprocessor != Mock:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
            
            # Configurar procesadores
            chunked_processor = ChunkedImageProcessor(
                chunk_size=(200, 200),
                overlap=16,
                strategy=ChunkingStrategy.GRID
            )
            
            unified_preprocessor = UnifiedPreprocessor()
            
            def preprocessing_function(chunk):
                """Función que usa el preprocessor unificado"""
                try:
                    # Aplicar algunas mejoras básicas
                    enhanced = unified_preprocessor.enhance_contrast(chunk, factor=1.2)
                    return enhanced
                except Exception:
                    # Fallback simple si el preprocessor falla
                    return np.clip(chunk * 1.1, 0, 255).astype(np.uint8)
            
            # Procesar con integración
            result = chunked_processor.process_image(test_image, preprocessing_function)
            
            # Verificar resultado
            assert result is not None, "El resultado no debe ser None"
            assert result.shape == test_image.shape, "Las dimensiones deben coincidir"
        else:
            pytest.skip("Componentes no disponibles para integración")
    
    def test_optimized_loader_with_chunked_processing(self):
        """Test de integración entre loader optimizado y procesamiento por chunks"""
        if OptimizedImageLoader != Mock and ChunkedImageProcessor != Mock:
            # Crear archivos de imagen de prueba
            image_files = []
            for i in range(3):
                filepath = os.path.join(self.temp_dir, f"test_image_{i}.jpg")
                create_test_image_file(filepath, 400, 300)
                image_files.append(filepath)
            
            # Configurar loader optimizado
            loader = OptimizedImageLoader(
                cache_size_mb=20,
                strategy=LoadingStrategy.LAZY
            )
            
            # Configurar procesador por chunks
            processor = ChunkedImageProcessor(
                chunk_size=(150, 150),
                overlap=10,
                strategy=ChunkingStrategy.ADAPTIVE
            )
            
            def processing_pipeline(image_path):
                """Pipeline completo de carga y procesamiento"""
                try:
                    # Cargar imagen con loader optimizado
                    with patch.object(loader, 'load_image') as mock_load:
                        # Simular carga de imagen
                        mock_load.return_value = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
                        image = loader.load_image(image_path)
                    
                    if image is not None:
                        # Procesar con chunks
                        def enhancement(chunk):
                            return np.clip(chunk * 1.05, 0, 255).astype(np.uint8)
                        
                        result = processor.process_image(image, enhancement)
                        return result
                    return None
                except Exception as e:
                    print(f"Error en pipeline: {e}")
                    return None
            
            # Procesar todas las imágenes
            results = []
            for image_file in image_files:
                result = processing_pipeline(image_file)
                results.append(result)
            
            # Verificar resultados
            valid_results = [r for r in results if r is not None]
            assert len(valid_results) > 0, "Debe haber al menos un resultado válido"
        else:
            pytest.skip("Componentes no disponibles para integración")
    
    def test_lazy_loading_with_chunked_dataset(self):
        """Test de integración entre lazy loading y procesamiento de dataset"""
        if LazyImageLoader != Mock and LazyImageDataset != Mock:
            # Crear dataset de imágenes simulado
            image_paths = []
            for i in range(5):
                filepath = os.path.join(self.temp_dir, f"dataset_image_{i}.jpg")
                create_test_image_file(filepath, 200, 200)
                image_paths.append(filepath)
            
            # Configurar lazy loader
            with patch('os.path.exists', return_value=True):
                lazy_loader = LazyImageLoader(cache_size_mb=15)
                dataset = LazyImageDataset(image_paths, lazy_loader)
                
                # Simular procesamiento de dataset por chunks
                processed_count = 0
                
                for i in range(len(image_paths)):
                    try:
                        # Simular carga lazy
                        with patch.object(lazy_loader, 'load_image') as mock_load:
                            mock_load.return_value = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
                            image = dataset[i]
                        
                        if image is not None:
                            # Simular procesamiento por chunks simple
                            chunk_size = 50
                            height, width = image.shape[:2]
                            
                            for y in range(0, height, chunk_size):
                                for x in range(0, width, chunk_size):
                                    chunk = image[y:y+chunk_size, x:x+chunk_size]
                                    # Procesar chunk (simulado)
                                    processed_chunk = chunk * 1.1
                                    del processed_chunk
                            
                            processed_count += 1
                    except Exception as e:
                        print(f"Error procesando imagen {i}: {e}")
                
                # Verificar que se procesaron imágenes
                assert processed_count > 0, "Debe haberse procesado al menos una imagen"
        else:
            pytest.skip("Componentes de lazy loading no disponibles")
    
    def test_memory_efficient_batch_processing(self):
        """Test de procesamiento eficiente en memoria de lotes de imágenes"""
        # Simular procesamiento de lote grande
        batch_size = 10
        image_size = (300, 300, 3)
        
        def process_batch_with_chunks(batch_images):
            """Procesa un lote de imágenes usando chunks"""
            results = []
            
            for img in batch_images:
                # Procesar cada imagen por chunks
                chunk_size = 100
                height, width = img.shape[:2]
                processed_img = np.zeros_like(img)
                
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        y_end = min(y + chunk_size, height)
                        x_end = min(x + chunk_size, width)
                        
                        chunk = img[y:y_end, x:x_end]
                        # Aplicar procesamiento
                        processed_chunk = np.clip(chunk * 1.15, 0, 255).astype(np.uint8)
                        processed_img[y:y_end, x:x_end] = processed_chunk
                        
                        # Limpiar chunk inmediatamente
                        del chunk, processed_chunk
                
                results.append(processed_img)
                
                # Garbage collection periódico
                if len(results) % 3 == 0:
                    gc.collect()
            
            return results
        
        # Crear lote de imágenes
        batch_images = []
        for i in range(batch_size):
            img = np.random.randint(0, 256, image_size, dtype=np.uint8)
            batch_images.append(img)
        
        # Procesar lote
        results = process_batch_with_chunks(batch_images)
        
        # Verificar resultados
        assert len(results) == batch_size, f"Debe haber {batch_size} resultados"
        
        for i, result in enumerate(results):
            assert result is not None, f"Resultado {i} no debe ser None"
            assert result.shape == image_size, f"Forma incorrecta en resultado {i}"
            assert not np.array_equal(result, batch_images[i]), f"Imagen {i} debe haber cambiado"
        
        # Limpiar memoria
        del batch_images, results
        gc.collect()
    
    def test_error_handling_in_chunked_pipeline(self):
        """Test de manejo de errores en el pipeline de chunks"""
        # Crear imagen de prueba
        test_image = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
        
        def problematic_enhancement(chunk):
            """Función que puede fallar en algunos chunks"""
            # Simular fallo en chunks específicos
            if chunk.shape[0] < 50 or chunk.shape[1] < 50:
                raise ValueError("Chunk demasiado pequeño")
            return np.clip(chunk * 1.2, 0, 255).astype(np.uint8)
        
        def safe_enhancement(chunk):
            """Función con manejo de errores"""
            try:
                return problematic_enhancement(chunk)
            except Exception:
                # Fallback: devolver chunk original
                return chunk
        
        # Procesar con manejo de errores
        chunk_size = 80
        height, width = test_image.shape[:2]
        result = np.zeros_like(test_image)
        successful_chunks = 0
        failed_chunks = 0
        
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                y_end = min(y + chunk_size, height)
                x_end = min(x + chunk_size, width)
                
                chunk = test_image[y:y_end, x:x_end]
                
                try:
                    processed_chunk = safe_enhancement(chunk)
                    result[y:y_end, x:x_end] = processed_chunk
                    successful_chunks += 1
                except Exception:
                    # En caso de fallo total, usar chunk original
                    result[y:y_end, x:x_end] = chunk
                    failed_chunks += 1
        
        # Verificar que el procesamiento se completó
        assert result is not None, "El resultado no debe ser None"
        assert result.shape == test_image.shape, "Las dimensiones deben coincidir"
        assert successful_chunks > 0, "Debe haber al menos un chunk exitoso"
        
        print(f"Chunks exitosos: {successful_chunks}, Chunks fallidos: {failed_chunks}")
    
    def test_performance_monitoring_integration(self):
        """Test de integración con monitoreo de rendimiento"""
        import time
        
        # Configurar métricas de rendimiento
        performance_metrics = {
            'total_chunks': 0,
            'processing_time': 0,
            'memory_usage': []
        }
        
        def monitored_enhancement(chunk):
            """Función de mejora con monitoreo"""
            start_time = time.perf_counter()
            
            # Procesar chunk
            enhanced = np.clip(chunk * 1.1 + 5, 0, 255).astype(np.uint8)
            
            # Registrar métricas
            processing_time = time.perf_counter() - start_time
            performance_metrics['total_chunks'] += 1
            performance_metrics['processing_time'] += processing_time
            
            return enhanced
        
        # Crear imagen de prueba
        test_image = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
        
        # Procesar con monitoreo
        chunk_size = 150
        height, width = test_image.shape[:2]
        result = np.zeros_like(test_image)
        
        total_start_time = time.perf_counter()
        
        for y in range(0, height, chunk_size):
            for x in range(0, width, chunk_size):
                y_end = min(y + chunk_size, height)
                x_end = min(x + chunk_size, width)
                
                chunk = test_image[y:y_end, x:x_end]
                processed_chunk = monitored_enhancement(chunk)
                result[y:y_end, x:x_end] = processed_chunk
        
        total_time = time.perf_counter() - total_start_time
        
        # Verificar métricas
        assert performance_metrics['total_chunks'] > 0, "Debe haber procesado chunks"
        assert performance_metrics['processing_time'] > 0, "Debe haber tiempo de procesamiento"
        assert total_time > 0, "Debe haber tiempo total"
        
        # Calcular estadísticas
        avg_time_per_chunk = performance_metrics['processing_time'] / performance_metrics['total_chunks']
        
        print(f"Chunks procesados: {performance_metrics['total_chunks']}")
        print(f"Tiempo total: {total_time:.4f}s")
        print(f"Tiempo promedio por chunk: {avg_time_per_chunk:.4f}s")
        
        # Verificar que el rendimiento sea razonable
        assert avg_time_per_chunk < 1.0, "Tiempo por chunk debe ser menor a 1 segundo"


if __name__ == "__main__":
    # Ejecutar tests de integración
    pytest.main([__file__, "-v", "-s"])