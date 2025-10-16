#!/usr/bin/env python3
"""
Tests de Integración Completos para el Sistema SIGeC-Balistica
============================================================

Este módulo contiene tests de integración que validan el funcionamiento
conjunto de todos los componentes del sistema, incluyendo:
- Pipeline completo de procesamiento
- Integración con deep learning
- Cumplimiento de estándares NIST
- Gestión de base de datos
- Interfaces de usuario

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import time
from typing import Dict, Any, List, Tuple

# Imports del sistema
try:
    from core.unified_pipeline import UnifiedPipeline, PipelineResult
    from config.unified_config import UnifiedConfig
    from utils.dependency_manager import DependencyManager
    from common.test_helpers import create_test_image, TestImageGenerator
    from database.database_manager import DatabaseManager
    from performance.enhanced_monitoring_system import PerformanceMonitor
except ImportError as e:
    pytest.skip(f"Dependencias no disponibles: {e}", allow_module_level=True)

class TestCompleteIntegration:
    """Tests de integración completa del sistema"""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Configura el entorno de pruebas"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_images_dir = self.temp_dir / "test_images"
        self.test_images_dir.mkdir(exist_ok=True)
        
        # Crear imágenes de prueba
        self.image_generator = TestImageGenerator()
        self.test_image1_path = self.test_images_dir / "test_bullet1.jpg"
        self.test_image2_path = self.test_images_dir / "test_bullet2.jpg"
        
        # Generar imágenes de prueba realistas
        self._create_test_images()
        
        # Configuración de prueba
        self.test_config = self._create_test_config()
        
        yield
        
        # Limpieza
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_images(self):
        """Crea imágenes de prueba realistas"""
        # Imagen 1: Bala con características distintivas
        image1 = self.image_generator.create_ballistic_image(
            width=800, height=600,
            features=['striations', 'firing_pin', 'breech_face'],
            noise_level=0.1
        )
        self.image_generator.save_image(image1, self.test_image1_path)
        
        # Imagen 2: Bala similar pero con variaciones
        image2 = self.image_generator.create_ballistic_image(
            width=800, height=600,
            features=['striations', 'firing_pin', 'breech_face'],
            noise_level=0.15,
            rotation=5  # Ligera rotación
        )
        self.image_generator.save_image(image2, self.test_image2_path)
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Crea configuración de prueba"""
        return {
            'pipeline': {
                'enable_preprocessing': True,
                'enable_quality_assessment': True,
                'enable_roi_detection': True,
                'enable_matching': True,
                'enable_deep_learning': True,
                'enable_nist_compliance': True
            },
            'preprocessing': {
                'resize_images': True,
                'target_size': (512, 512),
                'enhance_contrast': True,
                'denoise': True,
                'normalize': True
            },
            'quality_assessment': {
                'min_snr': 10.0,
                'min_contrast': 0.3,
                'min_sharpness': 0.5,
                'enable_nist_metrics': True
            },
            'matching': {
                'algorithm': 'unified',
                'feature_detector': 'sift',
                'max_features': 1000,
                'match_threshold': 0.7
            },
            'deep_learning': {
                'enable': True,
                'model_type': 'ballistic_cnn',
                'device': 'cpu',
                'batch_size': 1,
                'confidence_threshold': 0.8
            },
            'database': {
                'enable': True,
                'connection_string': f'sqlite:///{self.temp_dir}/test_db.sqlite'
            },
            'performance': {
                'enable_monitoring': True,
                'enable_caching': True,
                'cache_size': 100
            }
        }
    
    def test_complete_pipeline_execution(self):
        """Test del pipeline completo de procesamiento"""
        # Inicializar pipeline
        pipeline = UnifiedPipeline(self.test_config)
        
        # Verificar inicialización exitosa
        assert pipeline.is_initialized(), "Pipeline no se inicializó correctamente"
        
        # Ejecutar procesamiento completo
        result = pipeline.process_images(
            str(self.test_image1_path),
            str(self.test_image2_path)
        )
        
        # Validar resultado
        assert isinstance(result, PipelineResult), "Resultado no es del tipo esperado"
        assert result.success, f"Procesamiento falló: {result.error_message}"
        assert result.similarity_score >= 0.0, "Score de similitud inválido"
        assert result.quality_score >= 0.0, "Score de calidad inválido"
        assert result.processing_time > 0, "Tiempo de procesamiento inválido"
        
        # Validar componentes específicos
        assert result.preprocessing_successful, "Preprocesamiento falló"
        assert result.quality_assessment_passed, "Evaluación de calidad falló"
        assert result.roi_detected, "Detección de ROI falló"
        assert result.matching_successful, "Matching falló"
        
        print(f"✅ Pipeline completo ejecutado exitosamente")
        print(f"   Similitud: {result.similarity_score:.3f}")
        print(f"   Calidad: {result.quality_score:.3f}")
        print(f"   Tiempo: {result.processing_time:.3f}s")
    
    def test_deep_learning_integration(self):
        """Test de integración con deep learning"""
        if not DependencyManager.is_available('torch'):
            pytest.skip("PyTorch no disponible")
        
        # Configurar pipeline con DL habilitado
        config = self.test_config.copy()
        config['deep_learning']['enable'] = True
        
        pipeline = UnifiedPipeline(config)
        result = pipeline.process_images(
            str(self.test_image1_path),
            str(self.test_image2_path)
        )
        
        # Validar integración DL
        assert result.dl_features_extracted, "Características DL no extraídas"
        assert result.dl_feature_vector1 is not None, "Vector de características 1 nulo"
        assert result.dl_feature_vector2 is not None, "Vector de características 2 nulo"
        assert result.dl_similarity_score >= 0.0, "Score DL inválido"
        assert result.dl_model_used is not None, "Modelo DL no especificado"
        
        print(f"✅ Integración Deep Learning validada")
        print(f"   Score DL: {result.dl_similarity_score:.3f}")
        print(f"   Modelo: {result.dl_model_used}")
    
    def test_nist_compliance_validation(self):
        """Test de cumplimiento de estándares NIST"""
        # Pipeline con validación NIST estricta
        config = self.test_config.copy()
        config['quality_assessment']['enable_nist_metrics'] = True
        config['quality_assessment']['strict_compliance'] = True
        
        pipeline = UnifiedPipeline(config)
        result = pipeline.process_images(
            str(self.test_image1_path),
            str(self.test_image2_path)
        )
        
        # Validar métricas NIST
        assert result.nist_compliant, "No cumple estándares NIST"
        assert 'nist_metrics' in result.metadata, "Métricas NIST no disponibles"
        
        nist_metrics = result.metadata['nist_metrics']
        assert 'snr' in nist_metrics, "SNR no calculado"
        assert 'contrast' in nist_metrics, "Contraste no calculado"
        assert 'uniformity' in nist_metrics, "Uniformidad no calculada"
        assert 'sharpness' in nist_metrics, "Nitidez no calculada"
        
        print(f"✅ Cumplimiento NIST validado")
        print(f"   SNR: {nist_metrics['snr']:.2f}")
        print(f"   Contraste: {nist_metrics['contrast']:.3f}")
    
    def test_database_integration(self):
        """Test de integración con base de datos"""
        # Configurar pipeline con BD
        config = self.test_config.copy()
        config['database']['enable'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Procesar y almacenar
        result = pipeline.process_images(
            str(self.test_image1_path),
            str(self.test_image2_path)
        )
        
        # Validar almacenamiento
        assert result.case_id is not None, "ID de caso no asignado"
        assert result.stored_in_database, "Resultado no almacenado en BD"
        
        # Verificar recuperación
        db_manager = DatabaseManager(config['database'])
        retrieved_case = db_manager.get_case(result.case_id)
        
        assert retrieved_case is not None, "Caso no recuperado de BD"
        assert retrieved_case['similarity_score'] == result.similarity_score
        
        print(f"✅ Integración Base de Datos validada")
        print(f"   Caso ID: {result.case_id}")
    
    def test_performance_monitoring(self):
        """Test del sistema de monitoreo de performance"""
        # Pipeline con monitoreo habilitado
        config = self.test_config.copy()
        config['performance']['enable_monitoring'] = True
        
        with PerformanceMonitor() as monitor:
            pipeline = UnifiedPipeline(config)
            result = pipeline.process_images(
                str(self.test_image1_path),
                str(self.test_image2_path)
            )
        
        # Validar métricas de performance
        metrics = monitor.get_metrics()
        
        assert 'total_processing_time' in metrics
        assert 'memory_usage' in metrics
        assert 'cpu_usage' in metrics
        assert metrics['total_processing_time'] > 0
        
        print(f"✅ Monitoreo de Performance validado")
        print(f"   Tiempo total: {metrics['total_processing_time']:.3f}s")
        print(f"   Memoria: {metrics['memory_usage']:.1f}MB")
    
    def test_error_handling_and_fallbacks(self):
        """Test de manejo de errores y sistemas de respaldo"""
        # Configurar pipeline con componente que falle
        config = self.test_config.copy()
        
        with patch('core.unified_pipeline.UnifiedPipeline._perform_matching') as mock_matching:
            # Simular fallo en matching
            mock_matching.side_effect = Exception("Matching failed")
            
            pipeline = UnifiedPipeline(config)
            result = pipeline.process_images(
                str(self.test_image1_path),
                str(self.test_image2_path)
            )
            
            # Validar manejo de errores
            assert not result.matching_successful, "Debería haber fallado el matching"
            assert result.fallback_used, "Sistema de respaldo no activado"
            assert result.error_message is not None, "Mensaje de error no registrado"
        
        print(f"✅ Manejo de errores validado")
    
    def test_concurrent_processing(self):
        """Test de procesamiento concurrente"""
        import threading
        import concurrent.futures
        
        config = self.test_config.copy()
        results = []
        errors = []
        
        def process_pair(image_pair):
            try:
                pipeline = UnifiedPipeline(config)
                result = pipeline.process_images(image_pair[0], image_pair[1])
                return result
            except Exception as e:
                errors.append(e)
                return None
        
        # Crear múltiples pares de imágenes
        image_pairs = [
            (str(self.test_image1_path), str(self.test_image2_path))
            for _ in range(5)
        ]
        
        # Procesamiento concurrente
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_pair, pair) for pair in image_pairs]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Validar resultados
        assert len(errors) == 0, f"Errores en procesamiento concurrente: {errors}"
        assert all(r is not None and r.success for r in results), "Algunos procesamientos fallaron"
        
        print(f"✅ Procesamiento concurrente validado")
        print(f"   Procesos exitosos: {len(results)}")
    
    def test_memory_management(self):
        """Test de gestión de memoria"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = self.test_config.copy()
        config['performance']['enable_caching'] = True
        
        # Procesar múltiples veces
        for i in range(10):
            pipeline = UnifiedPipeline(config)
            result = pipeline.process_images(
                str(self.test_image1_path),
                str(self.test_image2_path)
            )
            
            # Forzar limpieza
            del pipeline
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Validar que no hay fuga de memoria significativa
        assert memory_increase < 100, f"Posible fuga de memoria: {memory_increase:.1f}MB"
        
        print(f"✅ Gestión de memoria validada")
        print(f"   Incremento de memoria: {memory_increase:.1f}MB")
    
    def test_configuration_validation(self):
        """Test de validación de configuración"""
        # Configuración inválida
        invalid_configs = [
            {'pipeline': {}},  # Configuración vacía
            {'pipeline': {'enable_preprocessing': 'invalid'}},  # Tipo incorrecto
            {'matching': {'algorithm': 'nonexistent'}},  # Algoritmo inexistente
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError, KeyError)):
                pipeline = UnifiedPipeline(invalid_config)
        
        print(f"✅ Validación de configuración exitosa")
    
    def test_api_compatibility(self):
        """Test de compatibilidad de APIs"""
        # Verificar que las interfaces principales no han cambiado
        pipeline = UnifiedPipeline(self.test_config)
        
        # Métodos principales deben existir
        assert hasattr(pipeline, 'process_images'), "Método process_images no existe"
        assert hasattr(pipeline, 'is_initialized'), "Método is_initialized no existe"
        assert hasattr(pipeline, 'get_capabilities'), "Método get_capabilities no existe"
        
        # Resultado debe tener campos esperados
        result = pipeline.process_images(
            str(self.test_image1_path),
            str(self.test_image2_path)
        )
        
        expected_fields = [
            'success', 'similarity_score', 'quality_score', 'processing_time',
            'preprocessing_successful', 'matching_successful', 'metadata'
        ]
        
        for field in expected_fields:
            assert hasattr(result, field), f"Campo {field} no existe en resultado"
        
        print(f"✅ Compatibilidad de API validada")

class TestEndToEndScenarios:
    """Tests de escenarios end-to-end"""
    
    @pytest.fixture(autouse=True)
    def setup_e2e_environment(self):
        """Configura entorno para tests E2E"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cases_dir = self.temp_dir / "cases"
        self.cases_dir.mkdir(exist_ok=True)
        
        # Crear casos de prueba realistas
        self._create_test_cases()
        
        yield
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_cases(self):
        """Crea casos de prueba realistas"""
        generator = TestImageGenerator()
        
        # Caso 1: Match positivo (misma arma)
        self.positive_case = {
            'image1': self.cases_dir / "case1_evidence.jpg",
            'image2': self.cases_dir / "case1_reference.jpg",
            'expected_match': True,
            'min_similarity': 0.7
        }
        
        # Caso 2: Match negativo (armas diferentes)
        self.negative_case = {
            'image1': self.cases_dir / "case2_evidence.jpg",
            'image2': self.cases_dir / "case2_reference.jpg",
            'expected_match': False,
            'max_similarity': 0.3
        }
        
        # Generar imágenes para casos
        self._generate_case_images(generator)
    
    def _generate_case_images(self, generator):
        """Genera imágenes para los casos de prueba"""
        # Caso positivo - misma arma
        base_features = ['striations', 'firing_pin', 'breech_face']
        
        image1 = generator.create_ballistic_image(
            width=800, height=600,
            features=base_features,
            noise_level=0.1
        )
        generator.save_image(image1, self.positive_case['image1'])
        
        image2 = generator.create_ballistic_image(
            width=800, height=600,
            features=base_features,
            noise_level=0.12,
            rotation=2  # Ligera variación
        )
        generator.save_image(image2, self.positive_case['image2'])
        
        # Caso negativo - armas diferentes
        image3 = generator.create_ballistic_image(
            width=800, height=600,
            features=['different_striations', 'different_pin'],
            noise_level=0.1
        )
        generator.save_image(image3, self.negative_case['image1'])
        
        image4 = generator.create_ballistic_image(
            width=800, height=600,
            features=['other_striations', 'other_pin'],
            noise_level=0.1
        )
        generator.save_image(image4, self.negative_case['image2'])
    
    def test_positive_match_scenario(self):
        """Test de escenario de match positivo"""
        config = {
            'pipeline': {'enable_all': True},
            'matching': {'algorithm': 'unified', 'threshold': 0.6}
        }
        
        pipeline = UnifiedPipeline(config)
        result = pipeline.process_images(
            str(self.positive_case['image1']),
            str(self.positive_case['image2'])
        )
        
        assert result.success, "Procesamiento falló"
        assert result.similarity_score >= self.positive_case['min_similarity'], \
            f"Similitud muy baja: {result.similarity_score}"
        assert result.match_conclusion == "MATCH", "Conclusión incorrecta"
        
        print(f"✅ Escenario match positivo validado")
        print(f"   Similitud: {result.similarity_score:.3f}")
    
    def test_negative_match_scenario(self):
        """Test de escenario de match negativo"""
        config = {
            'pipeline': {'enable_all': True},
            'matching': {'algorithm': 'unified', 'threshold': 0.6}
        }
        
        pipeline = UnifiedPipeline(config)
        result = pipeline.process_images(
            str(self.negative_case['image1']),
            str(self.negative_case['image2'])
        )
        
        assert result.success, "Procesamiento falló"
        assert result.similarity_score <= self.negative_case['max_similarity'], \
            f"Similitud muy alta: {result.similarity_score}"
        assert result.match_conclusion == "NO_MATCH", "Conclusión incorrecta"
        
        print(f"✅ Escenario match negativo validado")
        print(f"   Similitud: {result.similarity_score:.3f}")

if __name__ == "__main__":
    # Ejecutar tests si se llama directamente
    pytest.main([__file__, "-v", "--tb=short"])