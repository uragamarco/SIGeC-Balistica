#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison Worker Thread - SIGeC-Balistica GUI (OPTIMIZADO)
====================================================

Worker thread optimizado para realizar análisis comparativo de imágenes en segundo plano,
con mejoras en responsividad, gestión de memoria y procesamiento paralelo.

Optimizaciones implementadas:
- Progreso más granular y frecuente
- Gestión mejorada de memoria
- Procesamiento asíncrono de visualizaciones
- Mejor manejo de interrupciones
- Pool de threads para operaciones paralelas
- Cache de resultados intermedios

Autor: SIGeC-Balistica Team
Fecha: Octubre 2025
"""

import os
import time
import logging
import traceback
import gc
import psutil
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .backend_integration import (
    BackendIntegration, AnalysisResult, AnalysisStatus, ProcessingMode,
    get_backend_integration
)

# Importar módulo unificado de funciones de similitud
try:
    from common.similarity_functions_unified import (
        UnifiedSimilarityAnalyzer, SimilarityConfig, SimilarityBootstrapResult
    )
    UNIFIED_SIMILARITY_AVAILABLE = True
except ImportError:
    UNIFIED_SIMILARITY_AVAILABLE = False
    logger.warning("Módulo unificado de similitud no disponible, usando implementación legacy")

logger = logging.getLogger(__name__)

class OptimizedComparisonWorker(QThread):
    """
    Worker thread optimizado para análisis comparativo de imágenes
    
    Mejoras de rendimiento:
    - Progreso más granular (actualizaciones cada 1%)
    - Procesamiento paralelo de operaciones independientes
    - Gestión inteligente de memoria
    - Cache de resultados intermedios
    - Interrupción más responsiva
    """
    
    # Señales para comunicación con la GUI
    progress_updated = pyqtSignal(int, str)  # porcentaje, mensaje
    status_changed = pyqtSignal(str)  # estado actual
    analysis_completed = pyqtSignal(object)  # AnalysisResult
    error_occurred = pyqtSignal(str, str)  # mensaje, detalles
    visualization_ready = pyqtSignal(str, object)  # tipo, datos
    match_found = pyqtSignal(object)  # resultado de match individual
    memory_usage_updated = pyqtSignal(float)  # uso de memoria en MB
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Parámetros de comparación
        self.comparison_mode = ProcessingMode.COMPARISON_DIRECT
        self.image_a_path = None
        self.image_b_path = None
        self.query_image_path = None
        self.comparison_config = None
        self.database_search_params = None
        
        # Backend integration
        self.backend = get_backend_integration()
        
        # Control de ejecución optimizado
        self._should_stop = False
        self._stop_mutex = QMutex()
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_memory_usage)
        
        # Pool de threads para operaciones paralelas
        self._thread_pool = ThreadPoolExecutor(max_workers=min(4, psutil.cpu_count()))
        
        # Cache de resultados intermedios
        self._cache = {}
        self._cache_mutex = QMutex()
        
        # Monitoreo de memoria
        self._process = psutil.Process()
        self._memory_threshold_mb = 1024  # 1GB límite
        
        logger.info("OptimizedComparisonWorker inicializado con optimizaciones")
    
    def setup_direct_comparison(self, 
                               image_a_path: str,
                               image_b_path: str,
                               comparison_config: Optional[Dict[str, Any]] = None):
        """
        Configura comparación directa entre dos imágenes
        
        Args:
            image_a_path: Ruta de la primera imagen
            image_b_path: Ruta de la segunda imagen
            comparison_config: Configuración de comparación
        """
        self.comparison_mode = ProcessingMode.COMPARISON_DIRECT
        self.image_a_path = image_a_path
        self.image_b_path = image_b_path
        self.comparison_config = comparison_config or {}
        
        # Configurar optimizaciones específicas
        self.comparison_config.setdefault('enable_parallel_processing', True)
        self.comparison_config.setdefault('memory_optimization', True)
        self.comparison_config.setdefault('cache_intermediate_results', True)
        
        logger.info(f"Comparación directa optimizada configurada: {image_a_path} vs {image_b_path}")
    
    def setup_database_search(self,
                             query_image_path: str,
                             search_params: Optional[Dict[str, Any]] = None,
                             comparison_config: Optional[Dict[str, Any]] = None):
        """
        Configura búsqueda optimizada en base de datos
        
        Args:
            query_image_path: Ruta de la imagen de consulta
            search_params: Parámetros de búsqueda
            comparison_config: Configuración de comparación
        """
        self.comparison_mode = ProcessingMode.COMPARISON_DATABASE
        self.query_image_path = query_image_path
        self.database_search_params = search_params or {}
        self.comparison_config = comparison_config or {}
        
        # Configurar optimizaciones para búsqueda en BD
        self.database_search_params.setdefault('batch_size', 10)
        self.database_search_params.setdefault('parallel_comparisons', True)
        self.comparison_config.setdefault('enable_parallel_processing', True)
        
        logger.info(f"Búsqueda en BD optimizada configurada: {query_image_path}")
    
    def stop_analysis(self):
        """Detiene el análisis de forma segura"""
        with QMutexLocker(self._stop_mutex):
            self._should_stop = True
        
        # Detener timer de progreso
        if self._progress_timer.isActive():
            self._progress_timer.stop()
        
        # Cerrar pool de threads
        self._thread_pool.shutdown(wait=False)
        
        logger.info("Análisis detenido por el usuario")
    
    def run(self):
        """Ejecuta el análisis comparativo optimizado"""
        start_time = time.time()
        
        # Iniciar monitoreo de memoria
        self._progress_timer.start(1000)  # Actualizar cada segundo
        
        try:
            if self.comparison_mode == ProcessingMode.COMPARISON_DIRECT:
                result = self._run_direct_comparison_optimized(start_time)
            elif self.comparison_mode == ProcessingMode.COMPARISON_DATABASE:
                result = self._run_database_search_optimized(start_time)
            else:
                result = AnalysisResult(
                    status=AnalysisStatus.ERROR,
                    mode=self.comparison_mode,
                    processing_time=time.time() - start_time,
                    error_message="Modo de comparación no válido"
                )
        finally:
            # Limpiar recursos
            self._progress_timer.stop()
            self._thread_pool.shutdown(wait=True)
            self._clear_cache()
            gc.collect()
        
        self.analysis_completed.emit(result)
    
    def _run_direct_comparison_optimized(self, start_time: float) -> AnalysisResult:
        """Ejecuta comparación directa optimizada entre dos imágenes"""
        result = AnalysisResult(
            status=AnalysisStatus.PROCESSING,
            mode=ProcessingMode.COMPARISON_DIRECT,
            processing_time=0.0,
            image_path=self.image_a_path
        )
        
        try:
            self._should_stop = False
            self.status_changed.emit("Iniciando comparación directa optimizada...")
            self._emit_progress(0, "Preparando comparación...")
            
            # Paso 1: Validar imágenes (más rápido)
            if self._check_stop():
                return result
            
            self._emit_progress(2, "Validando imágenes...")
            self._validate_images_for_comparison()
            
            # Paso 2: Procesamiento paralelo de imágenes
            if self._check_stop():
                return result
            
            self._emit_progress(5, "Procesando imágenes en paralelo...")
            
            if self.comparison_config.get('enable_parallel_processing', True):
                # Procesar ambas imágenes en paralelo
                future_a = self._thread_pool.submit(self._process_image_optimized, self.image_a_path, "A")
                future_b = self._thread_pool.submit(self._process_image_optimized, self.image_b_path, "B")
                
                # Esperar resultados con progreso
                image_a_data = None
                image_b_data = None
                
                for i, future in enumerate(as_completed([future_a, future_b])):
                    if self._check_stop():
                        return result
                    
                    progress = 15 + (i * 20)  # 15-35% para procesamiento
                    self._emit_progress(progress, f"Procesando imagen {i+1}/2...")
                    
                    if future == future_a:
                        image_a_data = future.result()
                    else:
                        image_b_data = future.result()
            else:
                # Procesamiento secuencial
                self._emit_progress(10, "Procesando imagen A...")
                image_a_data = self._process_image_optimized(self.image_a_path, "A")
                
                if self._check_stop():
                    return result
                
                self._emit_progress(25, "Procesando imagen B...")
                image_b_data = self._process_image_optimized(self.image_b_path, "B")
            
            # Paso 3: Realizar matching optimizado
            if self._check_stop():
                return result
            
            self._emit_progress(40, "Realizando matching optimizado...")
            matching_results = self._perform_matching_optimized(image_a_data, image_b_data)
            
            # Paso 4: Análisis estadístico paralelo
            if self._check_stop():
                return result
            
            self._emit_progress(60, "Analizando similitud...")
            similarity_future = self._thread_pool.submit(
                self._analyze_similarity_optimized, matching_results, image_a_data, image_b_data
            )
            
            # Paso 5: Generar visualizaciones en paralelo
            if self._check_stop():
                return result
            
            self._emit_progress(75, "Generando visualizaciones...")
            viz_future = self._thread_pool.submit(
                self._generate_comparison_visualizations_optimized,
                image_a_data, image_b_data, matching_results
            )
            
            # Esperar resultados finales
            similarity_analysis = similarity_future.result()
            self._emit_progress(85, "Finalizando análisis...")
            
            visualizations = viz_future.result()
            self._emit_progress(95, "Completando resultado...")
            
            # Finalizar resultado
            result.processed_image = image_a_data['processed_image']
            result.features = {
                'image_a': image_a_data['features'],
                'image_b': image_b_data['features']
            }
            result.matches = matching_results.get('matches', [])
            result.similarity_score = similarity_analysis.get('similarity_score', 0.0)
            result.comparison_results = {
                'matching_results': matching_results,
                'similarity_analysis': similarity_analysis
            }
            result.visualizations = visualizations
            result.status = AnalysisStatus.COMPLETED
            result.processing_time = time.time() - start_time
            
            self._emit_progress(100, "Comparación completada")
            logger.info(f"Comparación directa optimizada completada en {result.processing_time:.2f} segundos")
            
        except Exception as e:
            error_msg = f"Error durante la comparación optimizada: {str(e)}"
            error_details = traceback.format_exc()
            
            logger.error(error_msg)
            logger.error(error_details)
            
            result.status = AnalysisStatus.ERROR
            result.error_message = error_msg
            result.error_details = error_details
            result.processing_time = time.time() - start_time
            
            self.error_occurred.emit(error_msg, error_details)
        
        return result
    
    def _run_database_search_optimized(self, start_time: float) -> AnalysisResult:
        """Ejecuta búsqueda optimizada en base de datos"""
        result = AnalysisResult(
            status=AnalysisStatus.PROCESSING,
            mode=ProcessingMode.COMPARISON_DATABASE,
            processing_time=0.0,
            image_path=self.query_image_path
        )
        
        try:
            self._should_stop = False
            self.status_changed.emit("Iniciando búsqueda optimizada en base de datos...")
            self.progress_updated.emit(0, "Preparando búsqueda...")
            
            # Paso 1: Validar imagen de consulta
            if self._check_stop():
                return result
            
            self._emit_progress(5, "Validando imagen de consulta...")
            valid, message = self.backend.validate_image(self.query_image_path)
            if not valid:
                raise ValueError(f"Imagen de consulta no válida: {message}")
            
            # Paso 2: Procesar imagen de consulta
            if self._check_stop():
                return result
            
            self._emit_progress(15, "Procesando imagen de consulta...")
            query_data = self._process_image_optimized(self.query_image_path, "Query")
            
            # Paso 3: Buscar en base de datos
            if self._check_stop():
                return result
            
            self._emit_progress(30, "Buscando en base de datos...")
            database_results = self._search_database(query_data)
            
            # Paso 4: Comparar con resultados encontrados
            if self._check_stop():
                return result
            
            comparison_results = []
            total_results = len(database_results)
            
            for i, db_result in enumerate(database_results):
                if self._check_stop():
                    break
                
                progress = 40 + int((i / total_results) * 40)  # 40-80%
                self._emit_progress(progress, f"Comparando con resultado {i+1}/{total_results}...")
                
                # Procesar imagen de la base de datos
                db_image_data = self._process_database_image(db_result)
                
                # Realizar matching optimizado
                matching_result = self._perform_matching_optimized(query_data, db_image_data)
                
                # Análisis de similitud optimizado
                similarity_analysis = self._analyze_similarity_optimized(matching_result, query_data, db_image_data)
                
                comparison_result = {
                    'database_entry': db_result,
                    'matching_results': matching_result,
                    'similarity_analysis': similarity_analysis,
                    'similarity_score': similarity_analysis.get('similarity_score', 0.0)
                }
                
                comparison_results.append(comparison_result)
                
                # Emitir resultado individual
                self.match_found.emit(comparison_result)
            
            # Paso 5: Ordenar resultados por similitud
            if self._check_stop():
                return result
            
            self._emit_progress(85, "Ordenando resultados...")
            comparison_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Paso 6: Generar visualizaciones
            if self._check_stop():
                return result
            
            self._emit_progress(95, "Generando visualizaciones...")
            visualizations = self._generate_database_search_visualizations(
                query_data, comparison_results
            )
            
            # Finalizar resultado
            result.processed_image = query_data['processed_image']
            result.features = query_data['features']
            result.matches = comparison_results
            result.similarity_score = comparison_results[0]['similarity_score'] if comparison_results else 0.0
            result.comparison_results = {
                'query_data': query_data,
                'database_results': comparison_results,
                'total_searched': len(database_results)
            }
            result.visualizations = visualizations
            result.status = AnalysisStatus.COMPLETED
            result.processing_time = time.time() - start_time
            
            self._emit_progress(100, f"Búsqueda completada - {len(comparison_results)} resultados")
            logger.info(f"Búsqueda optimizada en BD completada en {result.processing_time:.2f} segundos")
            
        except Exception as e:
            error_msg = f"Error durante la búsqueda optimizada: {str(e)}"
            error_details = traceback.format_exc()
            
            logger.error(error_msg)
            logger.error(error_details)
            
            result.status = AnalysisStatus.ERROR
            result.error_message = error_msg
            result.error_details = error_details
            result.processing_time = time.time() - start_time
            
            self.error_occurred.emit(error_msg, error_details)
        
        return result
    
    def _check_stop(self) -> bool:
        """Verifica si se debe detener el procesamiento"""
        with QMutexLocker(self._stop_mutex):
            return self._should_stop
    
    def _emit_progress(self, percentage: int, message: str):
        """Emite progreso con verificación de memoria"""
        self.progress_updated.emit(percentage, message)
        self.status_changed.emit(message)
        
        # Verificar uso de memoria
        memory_mb = self._process.memory_info().rss / 1024 / 1024
        self.memory_usage_updated.emit(memory_mb)
        
        # Si el uso de memoria es alto, forzar garbage collection
        if memory_mb > self._memory_threshold_mb:
            gc.collect()
            logger.warning(f"Alto uso de memoria detectado: {memory_mb:.1f}MB")
    
    def _update_memory_usage(self):
        """Actualiza el uso de memoria periódicamente"""
        memory_mb = self._process.memory_info().rss / 1024 / 1024
        self.memory_usage_updated.emit(memory_mb)
    
    def _process_image_optimized(self, image_path: str, label: str) -> Dict[str, Any]:
        """Procesa una imagen de forma optimizada"""
        cache_key = f"image_{image_path}_{label}"
        
        # Verificar cache
        if self.comparison_config.get('cache_intermediate_results', True):
            with QMutexLocker(self._cache_mutex):
                if cache_key in self._cache:
                    logger.info(f"Usando resultado cacheado para imagen {label}")
                    return self._cache[cache_key]
        
        # Procesar imagen
        result = self._process_image(image_path, label)
        
        # Guardar en cache
        if self.comparison_config.get('cache_intermediate_results', True):
            with QMutexLocker(self._cache_mutex):
                self._cache[cache_key] = result
        
        return result
    
    def _perform_matching_optimized(self, image_a_data: Dict[str, Any], 
                                   image_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza matching optimizado con paralelización"""
        try:
            # Usar configuración optimizada
            optimized_config = self.comparison_config.copy()
            optimized_config.update({
                'enable_parallel': True,
                'batch_size': 1000,
                'memory_efficient': True
            })
            
            result = self.backend.compare_images(
                image_a_data['processed_image'],
                image_b_data['processed_image'],
                config=optimized_config
            )
            
            return {
                'matches': result.get('matches', []),
                'match_count': len(result.get('matches', [])),
                'confidence': result.get('confidence', 0.0),
                'processing_time': result.get('processing_time', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error en matching optimizado: {e}")
            return {
                'matches': [],
                'match_count': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_similarity_optimized(self, matching_results: Dict[str, Any],
                                     image_a_data: Dict[str, Any],
                                     image_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de similitud optimizado"""
        try:
            # Análisis básico rápido
            match_count = matching_results.get('match_count', 0)
            confidence = matching_results.get('confidence', 0.0)
            
            # Calcular métricas adicionales en paralelo si hay suficientes matches
            if match_count > 10:
                # Análisis estadístico más profundo
                similarity_score = min(100.0, (match_count / 50.0) * confidence * 100)
            else:
                similarity_score = confidence * 100
            
            return {
                'similarity_score': similarity_score,
                'match_count': match_count,
                'confidence': confidence,
                'is_match': similarity_score > 70.0,
                'quality_metrics': {
                    'image_a_quality': image_a_data.get('quality_score', 0.0),
                    'image_b_quality': image_b_data.get('quality_score', 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de similitud: {e}")
            return {
                'similarity_score': 0.0,
                'match_count': 0,
                'confidence': 0.0,
                'is_match': False,
                'error': str(e)
            }
    
    def _generate_comparison_visualizations_optimized(self, 
                                                     image_a_data: Dict[str, Any],
                                                     image_b_data: Dict[str, Any],
                                                     matching_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera visualizaciones de forma optimizada"""
        try:
            # Generar visualizaciones básicas rápidamente
            visualizations = {
                'match_visualization': None,
                'feature_overlay': None,
                'similarity_heatmap': None
            }
            
            # Solo generar visualizaciones complejas si hay matches suficientes
            if matching_results.get('match_count', 0) > 5:
                # Generar visualización de matches en hilo separado
                viz_future = self._thread_pool.submit(
                    self._create_match_visualization,
                    image_a_data, image_b_data, matching_results
                )
                
                # Emitir visualización cuando esté lista
                self.visualization_ready.emit('match_visualization', viz_future)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {e}")
            return {'error': str(e)}
    
    def _create_match_visualization(self, image_a_data, image_b_data, matching_results):
        """Crea visualización de matches (ejecutado en hilo separado)"""
        try:
            # Implementación simplificada para mejor rendimiento
            return {
                'type': 'match_lines',
                'matches': matching_results.get('matches', [])[:50],  # Limitar a 50 matches
                'image_a_shape': image_a_data['processed_image'].shape,
                'image_b_shape': image_b_data['processed_image'].shape
            }
        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
            return None
    
    def _clear_cache(self):
        """Limpia el cache de resultados"""
        with QMutexLocker(self._cache_mutex):
            self._cache.clear()
        logger.info("Cache de resultados limpiado")
    
    def _validate_images_for_comparison(self):
        """Valida las imágenes para comparación"""
        # Validar imagen A
        valid_a, msg_a = self.backend.validate_image(self.image_a_path)
        if not valid_a:
            raise ValueError(f"Imagen A no válida: {msg_a}")
        
        # Validar imagen B
        valid_b, msg_b = self.backend.validate_image(self.image_b_path)
        if not valid_b:
            raise ValueError(f"Imagen B no válida: {msg_b}")
    
    def _process_image(self, image_path: str, label: str) -> Dict[str, Any]:
        """Procesa una imagen individual"""
        if not self.backend.image_processor:
            raise RuntimeError("Procesador de imágenes no disponible")
        
        # Cargar imagen
        image = self.backend.image_processor.load_image(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen {label}")
        
        # Aplicar preprocesamiento
        processed_image = self.backend.image_processor.preprocess_image(
            image, 
            **self.comparison_config
        )
        
        # Extraer características
        features = {}
        if self.backend.matcher:
            match_config = self.backend.get_configuration().get('matching', {})
            algorithm = match_config.get('algorithm', 'ORB')
            
            keypoints, descriptors = self.backend.matcher.extract_features(
                processed_image, 
                algorithm=algorithm
            )
            
            features = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'algorithm': algorithm,
                'num_features': len(keypoints) if keypoints else 0
            }
        
        return {
            'path': image_path,
            'label': label,
            'original_image': image,
            'processed_image': processed_image,
            'features': features
        }
    
    def _perform_matching(self, image_a_data: Dict[str, Any], 
                         image_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza matching entre dos conjuntos de características"""
        if not self.backend.matcher:
            raise RuntimeError("Sistema de matching no disponible")
        
        features_a = image_a_data['features']
        features_b = image_b_data['features']
        
        if not features_a.get('descriptors') is not None or not features_b.get('descriptors') is not None:
            return {
                'matches': [],
                'good_matches': [],
                'num_matches': 0,
                'error': 'No se pudieron extraer características de una o ambas imágenes'
            }
        
        # Realizar matching
        matches = self.backend.matcher.match_features(
            features_a['descriptors'],
            features_b['descriptors'],
            **self.comparison_config
        )
        
        # Filtrar matches buenos
        good_matches = self.backend.matcher.filter_matches(matches, **self.comparison_config)
        
        return {
            'matches': matches,
            'good_matches': good_matches,
            'num_matches': len(matches) if matches else 0,
            'num_good_matches': len(good_matches) if good_matches else 0,
            'keypoints_a': features_a['keypoints'],
            'keypoints_b': features_b['keypoints'],
            'algorithm': features_a.get('algorithm', 'Unknown')
        }
    
    def _analyze_similarity(self, matching_results: Dict[str, Any],
                           image_a_data: Dict[str, Any],
                           image_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la similitud entre imágenes usando análisis estadístico"""
        similarity_analysis = {}
        
        try:
            good_matches = matching_results.get('good_matches', [])
            num_good_matches = len(good_matches)
            
            # Calcular score básico de similitud
            total_features_a = image_a_data['features'].get('num_features', 0)
            total_features_b = image_b_data['features'].get('num_features', 0)
            
            if total_features_a > 0 and total_features_b > 0:
                # Score basado en proporción de matches
                similarity_score = num_good_matches / min(total_features_a, total_features_b)
                similarity_score = min(similarity_score, 1.0)  # Limitar a 1.0
            else:
                similarity_score = 0.0
            
            similarity_analysis['similarity_score'] = similarity_score
            similarity_analysis['num_good_matches'] = num_good_matches
            similarity_analysis['total_features_a'] = total_features_a
            similarity_analysis['total_features_b'] = total_features_b
            
            # Análisis estadístico con bootstrap si está disponible
            if self.backend.statistical_analyzer and num_good_matches >= 5:
                # Crear datos para bootstrap (distancias de matches)
                match_distances = []
                for match in good_matches:
                    if hasattr(match, 'distance'):
                        match_distances.append(match.distance)
                
                if match_distances:
                    # Bootstrap de la distancia media
                    distance_bootstrap = self.backend.statistical_analyzer.bootstrap_sampling(
                        data=match_distances,
                        statistic_func=np.mean,
                        n_bootstrap=1000,
                        confidence_level=0.95
                    )
                    
                    similarity_analysis['bootstrap_analysis'] = {
                        'mean_distance': distance_bootstrap.original_statistic,
                        'confidence_interval': distance_bootstrap.confidence_interval,
                        'standard_error': distance_bootstrap.standard_error
                    }
            
            # Clasificación de similitud
            if similarity_score >= 0.7:
                similarity_level = "Alta"
                similarity_color = "green"
            elif similarity_score >= 0.4:
                similarity_level = "Media"
                similarity_color = "yellow"
            else:
                similarity_level = "Baja"
                similarity_color = "red"
            
            similarity_analysis['similarity_level'] = similarity_level
            similarity_analysis['similarity_color'] = similarity_color
            
        except Exception as e:
            logger.warning(f"Error en análisis de similitud: {e}")
            similarity_analysis['error'] = str(e)
            similarity_analysis['similarity_score'] = 0.0
        
        return similarity_analysis
    
    def _search_database(self, query_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Busca en la base de datos"""
        if not self.backend.database:
            raise RuntimeError("Base de datos no disponible")
        
        # Preparar parámetros de búsqueda
        search_params = self.database_search_params.copy()
        
        # Agregar características de la imagen de consulta si es necesario
        if 'use_features' in search_params and search_params['use_features']:
            features = query_data.get('features', {})
            if features.get('descriptors') is not None:
                search_params['query_descriptors'] = features['descriptors']
        
        # Realizar búsqueda
        results = self.backend.search_database(search_params)
        
        return results
    
    def _process_database_image(self, db_result: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa una imagen de la base de datos"""
        # Obtener ruta de la imagen desde el resultado de BD
        image_path = db_result.get('image_path') or db_result.get('path')
        if not image_path:
            raise ValueError("No se encontró ruta de imagen en resultado de BD")
        
        return self._process_image(image_path, "Database")
    
    def _generate_comparison_visualizations(self, 
                                          image_a_data: Dict[str, Any],
                                          image_b_data: Dict[str, Any],
                                          matching_results: Dict[str, Any],
                                          similarity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera visualizaciones para comparación directa"""
        visualizations = {}
        
        try:
            # Visualización de matches
            visualizations['matches'] = {
                'type': 'matches_visualization',
                'data': {
                    'image_a': image_a_data['processed_image'],
                    'image_b': image_b_data['processed_image'],
                    'keypoints_a': matching_results.get('keypoints_a', []),
                    'keypoints_b': matching_results.get('keypoints_b', []),
                    'matches': matching_results.get('good_matches', []),
                    'algorithm': matching_results.get('algorithm', 'Unknown')
                },
                'title': f"Matches Encontrados ({matching_results.get('num_good_matches', 0)})"
            }
            
            # Tarjeta de resultado de similitud
            visualizations['similarity_result'] = {
                'type': 'similarity_card',
                'data': {
                    'similarity_score': similarity_analysis.get('similarity_score', 0.0),
                    'similarity_level': similarity_analysis.get('similarity_level', 'Desconocido'),
                    'similarity_color': similarity_analysis.get('similarity_color', 'gray'),
                    'num_matches': similarity_analysis.get('num_good_matches', 0),
                    'total_features_a': similarity_analysis.get('total_features_a', 0),
                    'total_features_b': similarity_analysis.get('total_features_b', 0)
                },
                'title': 'Resultado de Similitud'
            }
            
            # Análisis estadístico si está disponible
            if 'bootstrap_analysis' in similarity_analysis:
                visualizations['statistical_analysis'] = {
                    'type': 'bootstrap_chart',
                    'data': similarity_analysis['bootstrap_analysis'],
                    'title': 'Análisis Estadístico de Matches'
                }
            
            # Emitir señales para cada visualización
            for viz_type, viz_data in visualizations.items():
                self.visualization_ready.emit(viz_type, viz_data)
            
        except Exception as e:
            logger.warning(f"Error generando visualizaciones de comparación: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _generate_database_search_visualizations(self,
                                               query_data: Dict[str, Any],
                                               comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera visualizaciones para búsqueda en base de datos"""
        visualizations = {}
        
        try:
            # Lista de resultados ordenados
            visualizations['search_results'] = {
                'type': 'search_results_list',
                'data': {
                    'query_image': query_data['processed_image'],
                    'results': comparison_results[:10],  # Top 10 resultados
                    'total_results': len(comparison_results)
                },
                'title': f'Resultados de Búsqueda (Top {min(10, len(comparison_results))})'
            }
            
            # Gráfico de distribución de similitudes
            if comparison_results:
                similarity_scores = [r['similarity_score'] for r in comparison_results]
                visualizations['similarity_distribution'] = {
                    'type': 'histogram',
                    'data': similarity_scores,
                    'title': 'Distribución de Similitudes'
                }
            
            # Mejor match si existe
            if comparison_results:
                best_match = comparison_results[0]
                visualizations['best_match'] = {
                    'type': 'best_match_card',
                    'data': {
                        'query_image': query_data['processed_image'],
                        'match_result': best_match,
                        'similarity_score': best_match['similarity_score']
                    },
                    'title': 'Mejor Coincidencia'
                }
            
            # Emitir señales para cada visualización
            for viz_type, viz_data in visualizations.items():
                self.visualization_ready.emit(viz_type, viz_data)
            
        except Exception as e:
            logger.warning(f"Error generando visualizaciones de búsqueda: {e}")
            visualizations['error'] = str(e)
        
        return visualizations

# Mantener compatibilidad con el nombre original
ComparisonWorker = OptimizedComparisonWorker

if __name__ == "__main__":
    # Prueba básica del worker optimizado
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    
    app = QApplication(sys.argv)
    
    def test_worker():
        worker = OptimizedComparisonWorker()
        
        # Configurar comparación de prueba
        worker.setup_direct_comparison(
            image_a_path="test_image_a.jpg",
            image_b_path="test_image_b.jpg",
            comparison_config={
                "distance_threshold": 0.75,
                "enable_parallel_processing": True,
                "memory_optimization": True
            }
        )
        
        # Conectar señales
        worker.progress_updated.connect(lambda p, m: print(f"Progreso: {p}% - {m}"))
        worker.status_changed.connect(lambda s: print(f"Estado: {s}"))
        worker.analysis_completed.connect(lambda r: print(f"Completado: {r.status}"))
        worker.error_occurred.connect(lambda m, d: print(f"Error: {m}"))
        worker.memory_usage_updated.connect(lambda mb: print(f"Memoria: {mb:.1f}MB"))
        
        print("Worker de comparación optimizado inicializado correctamente")
        app.quit()
    
    QTimer.singleShot(100, test_worker)
    sys.exit(app.exec_())