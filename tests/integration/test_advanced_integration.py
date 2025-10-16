#!/usr/bin/env python3
"""
Tests de Integración Avanzados para el Sistema SIGeC-Balistica
==============================================================

Este módulo contiene tests de integración avanzados que validan
escenarios complejos y casos edge del sistema completo.

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
import time
import threading
import concurrent.futures
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional
import psutil
import gc

# Imports del sistema
try:
    from core.unified_pipeline import UnifiedPipeline, PipelineResult
    from config.unified_config import UnifiedConfig
    from utils.dependency_manager import DependencyManager
    from common.test_helpers import TestImageGenerator, create_test_image
    from database.database_manager import DatabaseManager
    from performance.enhanced_monitoring_system import PerformanceMonitor
    from nist_standards.quality_metrics import NISTQualityMetrics
    from matching.unified_matcher import UnifiedMatcher
    from image_processing.preprocessor import ImagePreprocessor
    from deep_learning.ballistic_cnn import BallisticCNN
except ImportError as e:
    pytest.skip(f"Dependencias no disponibles: {e}", allow_module_level=True)

class TestAdvancedIntegrationScenarios:
    """Tests de escenarios de integración avanzados"""
    
    @pytest.fixture(autouse=True)
    def setup_advanced_test_environment(self):
        """Configura entorno avanzado de pruebas"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.temp_dir / "advanced_test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Crear datasets de prueba complejos
        self.image_generator = TestImageGenerator()
        self._create_complex_test_datasets()
        
        # Configuración avanzada
        self.advanced_config = self._create_advanced_config()
        
        yield
        
        # Limpieza
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_complex_test_datasets(self):
        """Crea datasets complejos para pruebas avanzadas"""
        # Dataset 1: Variaciones de calidad
        self.quality_dataset = {
            'high_quality': [],
            'medium_quality': [],
            'low_quality': [],
            'corrupted': []
        }
        
        # Dataset 2: Diferentes tipos de munición
        self.ammunition_dataset = {
            'pistol_9mm': [],
            'rifle_556': [],
            'shotgun_12ga': [],
            'revolver_38': []
        }
        
        # Dataset 3: Condiciones de iluminación
        self.lighting_dataset = {
            'optimal': [],
            'low_light': [],
            'high_contrast': [],
            'uneven': []
        }
        
        self._generate_quality_variations()
        self._generate_ammunition_types()
        self._generate_lighting_conditions()
    
    def _generate_quality_variations(self):
        """Genera imágenes con diferentes niveles de calidad"""
        base_features = ['striations', 'firing_pin', 'breech_face']
        
        # Alta calidad
        for i in range(5):
            img_path = self.test_data_dir / f"high_quality_{i}.jpg"
            image = self.image_generator.create_ballistic_image(
                width=1024, height=768,
                features=base_features,
                noise_level=0.02,
                blur_level=0.0,
                quality='high'
            )
            self.image_generator.save_image(image, img_path)
            self.quality_dataset['high_quality'].append(str(img_path))
        
        # Calidad media
        for i in range(5):
            img_path = self.test_data_dir / f"medium_quality_{i}.jpg"
            image = self.image_generator.create_ballistic_image(
                width=512, height=384,
                features=base_features,
                noise_level=0.1,
                blur_level=0.5,
                quality='medium'
            )
            self.image_generator.save_image(image, img_path)
            self.quality_dataset['medium_quality'].append(str(img_path))
        
        # Baja calidad
        for i in range(5):
            img_path = self.test_data_dir / f"low_quality_{i}.jpg"
            image = self.image_generator.create_ballistic_image(
                width=256, height=192,
                features=base_features,
                noise_level=0.3,
                blur_level=1.0,
                quality='low'
            )
            self.image_generator.save_image(image, img_path)
            self.quality_dataset['low_quality'].append(str(img_path))
    
    def _generate_ammunition_types(self):
        """Genera imágenes de diferentes tipos de munición"""
        ammunition_configs = {
            'pistol_9mm': {
                'features': ['fine_striations', 'circular_firing_pin', 'rectangular_breech'],
                'size': (600, 450)
            },
            'rifle_556': {
                'features': ['deep_striations', 'elongated_firing_pin', 'square_breech'],
                'size': (800, 600)
            },
            'shotgun_12ga': {
                'features': ['wide_striations', 'large_firing_pin', 'circular_breech'],
                'size': (700, 525)
            },
            'revolver_38': {
                'features': ['curved_striations', 'small_firing_pin', 'hexagonal_breech'],
                'size': (550, 400)
            }
        }
        
        for ammo_type, config in ammunition_configs.items():
            for i in range(3):
                img_path = self.test_data_dir / f"{ammo_type}_{i}.jpg"
                image = self.image_generator.create_ballistic_image(
                    width=config['size'][0],
                    height=config['size'][1],
                    features=config['features'],
                    noise_level=0.08
                )
                self.image_generator.save_image(image, img_path)
                self.ammunition_dataset[ammo_type].append(str(img_path))
    
    def _generate_lighting_conditions(self):
        """Genera imágenes con diferentes condiciones de iluminación"""
        lighting_configs = {
            'optimal': {'brightness': 1.0, 'contrast': 1.0},
            'low_light': {'brightness': 0.3, 'contrast': 0.8},
            'high_contrast': {'brightness': 1.2, 'contrast': 1.5},
            'uneven': {'brightness': 0.8, 'contrast': 1.3, 'gradient': True}
        }
        
        base_features = ['striations', 'firing_pin', 'breech_face']
        
        for condition, config in lighting_configs.items():
            for i in range(3):
                img_path = self.test_data_dir / f"lighting_{condition}_{i}.jpg"
                image = self.image_generator.create_ballistic_image(
                    width=640, height=480,
                    features=base_features,
                    **config
                )
                self.image_generator.save_image(image, img_path)
                self.lighting_dataset[condition].append(str(img_path))
    
    def _create_advanced_config(self) -> Dict[str, Any]:
        """Crea configuración avanzada para pruebas"""
        return {
            'pipeline': {
                'enable_all': True,
                'adaptive_processing': True,
                'quality_based_optimization': True,
                'multi_algorithm_fusion': True
            },
            'preprocessing': {
                'adaptive_enhancement': True,
                'noise_reduction_level': 'aggressive',
                'contrast_optimization': True,
                'illumination_correction': True
            },
            'quality_assessment': {
                'multi_metric_evaluation': True,
                'adaptive_thresholds': True,
                'nist_compliance_strict': True,
                'quality_prediction': True
            },
            'matching': {
                'multi_algorithm_ensemble': True,
                'adaptive_feature_selection': True,
                'confidence_weighting': True,
                'cross_validation': True
            },
            'deep_learning': {
                'enable_ensemble': True,
                'model_fusion': True,
                'uncertainty_estimation': True,
                'active_learning': True
            },
            'performance': {
                'enable_gpu_acceleration': True,
                'memory_optimization': True,
                'parallel_processing': True,
                'caching_strategy': 'intelligent'
            },
            'database': {
                'enable_indexing': True,
                'similarity_search': True,
                'metadata_extraction': True,
                'backup_strategy': 'incremental'
            }
        }
    
    def test_multi_quality_processing(self):
        """Test de procesamiento con múltiples niveles de calidad"""
        pipeline = UnifiedPipeline(self.advanced_config)
        
        quality_results = {}
        
        # Procesar cada nivel de calidad
        for quality_level, images in self.quality_dataset.items():
            if len(images) >= 2:  # Necesitamos al menos 2 imágenes para comparar
                result = pipeline.process_images(images[0], images[1])
                quality_results[quality_level] = result
                
                # Validar que el pipeline se adapta a la calidad
                assert result.success, f"Procesamiento falló para calidad {quality_level}"
                
                # Verificar adaptación de parámetros
                if quality_level == 'high_quality':
                    assert result.processing_parameters['enhancement_level'] == 'minimal'
                elif quality_level == 'low_quality':
                    assert result.processing_parameters['enhancement_level'] == 'aggressive'
        
        # Comparar resultados entre calidades
        if 'high_quality' in quality_results and 'low_quality' in quality_results:
            high_q_result = quality_results['high_quality']
            low_q_result = quality_results['low_quality']
            
            # La calidad alta debería tener mejor score de calidad
            assert high_q_result.quality_score > low_q_result.quality_score
            
            # El tiempo de procesamiento puede variar
            assert high_q_result.processing_time > 0
            assert low_q_result.processing_time > 0
        
        print("✅ Procesamiento multi-calidad validado")
        for quality, result in quality_results.items():
            print(f"   {quality}: Score={result.quality_score:.3f}, Tiempo={result.processing_time:.3f}s")
    
    def test_ammunition_type_classification(self):
        """Test de clasificación por tipo de munición"""
        pipeline = UnifiedPipeline(self.advanced_config)
        
        classification_results = {}
        
        # Procesar cada tipo de munición
        for ammo_type, images in self.ammunition_dataset.items():
            if len(images) >= 2:
                result = pipeline.process_images(images[0], images[1])
                classification_results[ammo_type] = result
                
                # Validar clasificación
                assert result.success, f"Procesamiento falló para {ammo_type}"
                assert 'ammunition_type' in result.metadata, "Tipo de munición no clasificado"
                
                # Verificar que se detectó el tipo correcto
                detected_type = result.metadata['ammunition_type']
                assert ammo_type.split('_')[0] in detected_type.lower(), \
                    f"Tipo incorrecto detectado: {detected_type} vs {ammo_type}"
        
        # Validar diferenciación entre tipos
        similarity_matrix = {}
        ammo_types = list(classification_results.keys())
        
        for i, type1 in enumerate(ammo_types):
            for j, type2 in enumerate(ammo_types[i+1:], i+1):
                # Comparar entre tipos diferentes
                result = pipeline.process_images(
                    self.ammunition_dataset[type1][0],
                    self.ammunition_dataset[type2][0]
                )
                
                similarity_matrix[f"{type1}_vs_{type2}"] = result.similarity_score
                
                # Diferentes tipos deberían tener menor similitud
                assert result.similarity_score < 0.7, \
                    f"Similitud muy alta entre tipos diferentes: {result.similarity_score}"
        
        print("✅ Clasificación por tipo de munición validada")
        for comparison, similarity in similarity_matrix.items():
            print(f"   {comparison}: {similarity:.3f}")
    
    def test_lighting_condition_robustness(self):
        """Test de robustez ante diferentes condiciones de iluminación"""
        pipeline = UnifiedPipeline(self.advanced_config)
        
        lighting_results = {}
        
        # Procesar cada condición de iluminación
        for condition, images in self.lighting_dataset.items():
            if len(images) >= 2:
                result = pipeline.process_images(images[0], images[1])
                lighting_results[condition] = result
                
                assert result.success, f"Procesamiento falló para condición {condition}"
                
                # Verificar corrección de iluminación
                if condition != 'optimal':
                    assert result.preprocessing_applied['illumination_correction'], \
                        f"Corrección de iluminación no aplicada para {condition}"
        
        # Comparar robustez
        if 'optimal' in lighting_results:
            optimal_score = lighting_results['optimal'].similarity_score
            
            for condition, result in lighting_results.items():
                if condition != 'optimal':
                    # La diferencia no debería ser muy grande tras corrección
                    score_diff = abs(optimal_score - result.similarity_score)
                    assert score_diff < 0.3, \
                        f"Gran diferencia con condición {condition}: {score_diff}"
        
        print("✅ Robustez ante condiciones de iluminación validada")
        for condition, result in lighting_results.items():
            print(f"   {condition}: Score={result.similarity_score:.3f}")
    
    def test_ensemble_model_performance(self):
        """Test de rendimiento de modelos ensemble"""
        config = self.advanced_config.copy()
        config['deep_learning']['enable_ensemble'] = True
        config['matching']['multi_algorithm_ensemble'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Usar imágenes de alta calidad para test de ensemble
        high_quality_images = self.quality_dataset['high_quality']
        
        if len(high_quality_images) >= 2:
            result = pipeline.process_images(
                high_quality_images[0],
                high_quality_images[1]
            )
            
            assert result.success, "Procesamiento ensemble falló"
            
            # Verificar que se usaron múltiples modelos
            assert 'ensemble_models_used' in result.metadata, "Información de ensemble no disponible"
            models_used = result.metadata['ensemble_models_used']
            assert len(models_used) > 1, "Ensemble no usó múltiples modelos"
            
            # Verificar métricas de confianza
            assert 'ensemble_confidence' in result.metadata, "Confianza de ensemble no disponible"
            ensemble_confidence = result.metadata['ensemble_confidence']
            assert 0.0 <= ensemble_confidence <= 1.0, "Confianza de ensemble fuera de rango"
            
            # Verificar que el ensemble mejora la precisión
            assert result.confidence_score > 0.8, "Confianza de ensemble muy baja"
        
        print("✅ Rendimiento de modelos ensemble validado")
        print(f"   Modelos usados: {len(models_used)}")
        print(f"   Confianza ensemble: {ensemble_confidence:.3f}")
    
    def test_adaptive_parameter_optimization(self):
        """Test de optimización adaptiva de parámetros"""
        config = self.advanced_config.copy()
        config['pipeline']['adaptive_processing'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Procesar imágenes con diferentes características
        test_pairs = [
            (self.quality_dataset['high_quality'][0], self.quality_dataset['high_quality'][1]),
            (self.quality_dataset['low_quality'][0], self.quality_dataset['low_quality'][1]),
            (self.lighting_dataset['optimal'][0], self.lighting_dataset['low_light'][0])
        ]
        
        adaptive_results = []
        
        for i, (img1, img2) in enumerate(test_pairs):
            result = pipeline.process_images(img1, img2)
            adaptive_results.append(result)
            
            assert result.success, f"Procesamiento adaptivo falló para par {i}"
            
            # Verificar que se aplicó optimización adaptiva
            assert 'adaptive_parameters' in result.metadata, "Parámetros adaptativos no disponibles"
            adaptive_params = result.metadata['adaptive_parameters']
            
            # Verificar que los parámetros se ajustaron
            assert 'preprocessing_level' in adaptive_params, "Nivel de preprocesamiento no adaptado"
            assert 'matching_threshold' in adaptive_params, "Umbral de matching no adaptado"
        
        # Verificar que los parámetros son diferentes para diferentes tipos de imágenes
        param_sets = [result.metadata['adaptive_parameters'] for result in adaptive_results]
        
        # Al menos algunos parámetros deberían ser diferentes
        preprocessing_levels = [params['preprocessing_level'] for params in param_sets]
        assert len(set(preprocessing_levels)) > 1, "Parámetros no se adaptaron correctamente"
        
        print("✅ Optimización adaptiva de parámetros validada")
        for i, params in enumerate(param_sets):
            print(f"   Par {i}: Preprocesamiento={params['preprocessing_level']}, "
                  f"Umbral={params['matching_threshold']:.3f}")
    
    def test_large_scale_batch_processing(self):
        """Test de procesamiento por lotes a gran escala"""
        # Crear un lote grande de imágenes
        batch_size = 20
        batch_images = []
        
        # Usar todas las imágenes disponibles
        all_images = []
        for dataset in [self.quality_dataset, self.ammunition_dataset, self.lighting_dataset]:
            for images in dataset.values():
                all_images.extend(images)
        
        # Crear pares para el lote
        for i in range(0, min(batch_size * 2, len(all_images)), 2):
            if i + 1 < len(all_images):
                batch_images.append((all_images[i], all_images[i + 1]))
        
        config = self.advanced_config.copy()
        config['performance']['parallel_processing'] = True
        config['performance']['batch_optimization'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Procesar lote
        start_time = time.time()
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(pipeline.process_images, img1, img2)
                for img1, img2 in batch_images
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error en procesamiento de lote: {e}")
        
        total_time = time.time() - start_time
        
        # Validar resultados
        successful_results = [r for r in batch_results if r.success]
        success_rate = len(successful_results) / len(batch_images)
        
        assert success_rate > 0.8, f"Tasa de éxito muy baja: {success_rate:.2f}"
        assert total_time < len(batch_images) * 10, "Procesamiento por lotes muy lento"
        
        # Calcular estadísticas
        similarity_scores = [r.similarity_score for r in successful_results]
        avg_similarity = np.mean(similarity_scores)
        std_similarity = np.std(similarity_scores)
        
        print("✅ Procesamiento por lotes a gran escala validado")
        print(f"   Pares procesados: {len(batch_images)}")
        print(f"   Tasa de éxito: {success_rate:.2%}")
        print(f"   Tiempo total: {total_time:.2f}s")
        print(f"   Similitud promedio: {avg_similarity:.3f} ± {std_similarity:.3f}")
    
    def test_memory_efficiency_large_images(self):
        """Test de eficiencia de memoria con imágenes grandes"""
        # Crear imágenes muy grandes
        large_images = []
        for i in range(3):
            img_path = self.test_data_dir / f"large_image_{i}.jpg"
            large_image = self.image_generator.create_ballistic_image(
                width=2048, height=1536,  # Imagen grande
                features=['detailed_striations', 'firing_pin', 'breech_face'],
                noise_level=0.05
            )
            self.image_generator.save_image(large_image, img_path)
            large_images.append(str(img_path))
        
        config = self.advanced_config.copy()
        config['performance']['memory_optimization'] = True
        config['performance']['streaming_processing'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Monitorear memoria
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Procesar imágenes grandes
        result = pipeline.process_images(large_images[0], large_images[1])
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Forzar limpieza
        del pipeline
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_cleanup = peak_memory - final_memory
        
        # Validar eficiencia de memoria
        assert result.success, "Procesamiento de imágenes grandes falló"
        assert memory_increase < 500, f"Uso excesivo de memoria: {memory_increase:.1f}MB"
        assert memory_cleanup > memory_increase * 0.7, "Limpieza de memoria insuficiente"
        
        print("✅ Eficiencia de memoria con imágenes grandes validada")
        print(f"   Incremento de memoria: {memory_increase:.1f}MB")
        print(f"   Limpieza de memoria: {memory_cleanup:.1f}MB")
    
    def test_error_recovery_and_fallbacks(self):
        """Test de recuperación de errores y sistemas de respaldo"""
        config = self.advanced_config.copy()
        config['pipeline']['enable_fallbacks'] = True
        config['pipeline']['error_recovery'] = True
        
        # Test con diferentes tipos de fallos simulados
        failure_scenarios = [
            ('preprocessing_failure', 'image_processing.preprocessor.ImagePreprocessor.preprocess'),
            ('matching_failure', 'matching.unified_matcher.UnifiedMatcher.match_features'),
            ('dl_failure', 'deep_learning.ballistic_cnn.BallisticCNN.extract_features'),
            ('quality_failure', 'nist_standards.quality_metrics.NISTQualityMetrics.calculate_metrics')
        ]
        
        recovery_results = {}
        
        for scenario_name, method_to_patch in failure_scenarios:
            with patch(method_to_patch, side_effect=Exception(f"Simulated {scenario_name}")):
                pipeline = UnifiedPipeline(config)
                
                result = pipeline.process_images(
                    self.quality_dataset['high_quality'][0],
                    self.quality_dataset['high_quality'][1]
                )
                
                recovery_results[scenario_name] = result
                
                # El sistema debería recuperarse usando fallbacks
                assert result.fallback_used, f"Fallback no usado para {scenario_name}"
                assert result.error_message is not None, f"Error no registrado para {scenario_name}"
                
                # Debería haber algún resultado, aunque degradado
                if scenario_name != 'preprocessing_failure':  # Preprocessing es crítico
                    assert result.similarity_score >= 0, f"Score inválido para {scenario_name}"
        
        print("✅ Recuperación de errores y fallbacks validada")
        for scenario, result in recovery_results.items():
            print(f"   {scenario}: Fallback={result.fallback_used}, "
                  f"Score={result.similarity_score:.3f}")
    
    def test_cross_validation_consistency(self):
        """Test de consistencia mediante validación cruzada"""
        config = self.advanced_config.copy()
        config['matching']['cross_validation'] = True
        config['matching']['consistency_check'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Usar el mismo par de imágenes múltiples veces
        test_image1 = self.quality_dataset['high_quality'][0]
        test_image2 = self.quality_dataset['high_quality'][1]
        
        # Ejecutar múltiples veces
        results = []
        for i in range(5):
            result = pipeline.process_images(test_image1, test_image2)
            results.append(result)
            assert result.success, f"Ejecución {i} falló"
        
        # Analizar consistencia
        similarity_scores = [r.similarity_score for r in results]
        quality_scores = [r.quality_score for r in results]
        
        # Calcular variabilidad
        similarity_std = np.std(similarity_scores)
        quality_std = np.std(quality_scores)
        
        # La variabilidad debería ser baja
        assert similarity_std < 0.05, f"Variabilidad de similitud muy alta: {similarity_std:.4f}"
        assert quality_std < 0.05, f"Variabilidad de calidad muy alta: {quality_std:.4f}"
        
        # Verificar validación cruzada
        for result in results:
            assert 'cross_validation_score' in result.metadata, "Score de validación cruzada no disponible"
            cv_score = result.metadata['cross_validation_score']
            assert 0.0 <= cv_score <= 1.0, "Score de validación cruzada fuera de rango"
        
        print("✅ Consistencia mediante validación cruzada validada")
        print(f"   Variabilidad similitud: {similarity_std:.4f}")
        print(f"   Variabilidad calidad: {quality_std:.4f}")
        print(f"   Score CV promedio: {np.mean([r.metadata['cross_validation_score'] for r in results]):.3f}")

if __name__ == "__main__":
    # Ejecutar tests si se llama directamente
    pytest.main([__file__, "-v", "--tb=short"])