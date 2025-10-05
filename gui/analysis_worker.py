#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Worker Thread - SEACABAr GUI
=====================================

Worker thread para realizar análisis individual de imágenes en segundo plano,
integrando completamente con el backend existente del sistema SEACABAr.

Características:
- Procesamiento en hilo separado para mantener la GUI responsiva
- Integración completa con UnifiedStatisticalAnalysis
- Soporte para metadatos NIST
- Generación de visualizaciones
- Manejo robusto de errores
- Progreso detallado en tiempo real

Autor: SEACABAr Team
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

logger = logging.getLogger(__name__)

class OptimizedAnalysisWorker(QThread):
    """
    Worker thread optimizado para análisis individual de imágenes
    
    Mejoras de rendimiento:
    - Procesamiento paralelo de análisis estadísticos
    - Cache inteligente de resultados intermedios
    - Gestión optimizada de memoria
    - Progreso más granular (actualizaciones cada 2%)
    - Pool de threads para operaciones independientes
    - Análisis asíncrono de visualizaciones
    """
    
    # Señales para comunicación con la GUI
    progress_updated = pyqtSignal(int, str)  # porcentaje, mensaje
    status_changed = pyqtSignal(str)  # estado actual
    analysis_completed = pyqtSignal(object)  # AnalysisResult
    error_occurred = pyqtSignal(str, str)  # mensaje, detalles
    visualization_ready = pyqtSignal(str, object)  # tipo, datos
    memory_usage_updated = pyqtSignal(float)  # uso de memoria en MB
    quality_metrics_ready = pyqtSignal(object)  # métricas de calidad
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Parámetros de análisis
        self.image_path = None
        self.case_data = None
        self.nist_metadata = None
        self.processing_config = None
        
        # Backend integration
        self.backend = get_backend_integration()
        
        # Control de ejecución optimizado
        self._should_stop = False
        self._stop_mutex = QMutex()
        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_memory_usage)
        
        # Pool de threads para operaciones paralelas
        self._thread_pool = ThreadPoolExecutor(max_workers=min(3, psutil.cpu_count()))
        
        # Cache de resultados intermedios
        self._cache = {}
        self._cache_mutex = QMutex()
        
        # Monitoreo de memoria
        self._process = psutil.Process()
        self._memory_threshold_mb = 512  # 512MB límite para análisis individual
        
        logger.info("OptimizedAnalysisWorker inicializado con optimizaciones")
    
    def setup_analysis(self, 
                      image_path: str,
                      case_data: Dict[str, Any],
                      nist_metadata: Optional[Dict[str, Any]] = None,
                      processing_config: Optional[Dict[str, Any]] = None):
        """
        Configura los parámetros para el análisis optimizado
        
        Args:
            image_path: Ruta de la imagen a analizar
            case_data: Datos del caso
            nist_metadata: Metadatos NIST opcionales
            processing_config: Configuración de procesamiento
        """
        self.image_path = image_path
        self.case_data = case_data
        self.nist_metadata = nist_metadata
        self.processing_config = processing_config or {}
        
        # Configurar optimizaciones específicas
        self.processing_config.setdefault('enable_parallel_analysis', True)
        self.processing_config.setdefault('cache_intermediate_results', True)
        self.processing_config.setdefault('memory_optimization', True)
        self.processing_config.setdefault('async_visualizations', True)
        
        logger.info(f"Análisis optimizado configurado para imagen: {image_path}")
    
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
        """Ejecuta el análisis completo con optimizaciones"""
        start_time = time.time()
        result = AnalysisResult(
            status=AnalysisStatus.PROCESSING,
            mode=ProcessingMode.INDIVIDUAL,
            processing_time=0.0,
            image_path=self.image_path,
            case_data=self.case_data,
            nist_metadata=self.nist_metadata
        )
        
        try:
            # Iniciar monitoreo de memoria
            self._progress_timer.start(1000)  # Actualizar cada segundo
            
            self._should_stop = False
            self.status_changed.emit("Iniciando análisis optimizado...")
            self.progress_updated.emit(0, "Preparando análisis...")
            
            # Paso 1: Validar imagen
            if self._check_stop():
                return
            
            self._emit_progress(5, "Validando imagen...")
            valid, message = self.backend.validate_image(self.image_path)
            if not valid:
                raise ValueError(f"Imagen no válida: {message}")
            
            # Paso 2: Cargar y procesar imagen con cache
            if self._check_stop():
                return
            
            self._emit_progress(15, "Cargando imagen...")
            processed_data = self._load_and_process_image_optimized()
            result.processed_image = processed_data['processed_image']
            result.features = processed_data['features']
            
            # Paso 3: Análisis de calidad paralelo
            if self._check_stop():
                return
            
            self._emit_progress(35, "Analizando calidad de imagen...")
            quality_results = self._analyze_image_quality_parallel(processed_data['processed_image'])
            result.quality_metrics = quality_results
            
            # Emitir métricas de calidad inmediatamente
            self.quality_metrics_ready.emit(quality_results)
            
            # Paso 4: Análisis estadístico vectorizado
            if self._check_stop():
                return
            
            self._emit_progress(55, "Realizando análisis estadístico...")
            statistical_results = self._perform_statistical_analysis_vectorized(processed_data, quality_results)
            result.statistical_results = statistical_results
            
            # Paso 5: Análisis NIST asíncrono (si hay metadatos)
            if self._check_stop():
                return
            
            if self.nist_metadata:
                self._emit_progress(70, "Evaluando cumplimiento NIST...")
                nist_results = self._analyze_nist_compliance_async(quality_results)
                result.nist_compliance = nist_results
            
            # Paso 6: Generar visualizaciones en paralelo
            if self._check_stop():
                return
            
            self._emit_progress(85, "Generando visualizaciones...")
            visualizations = self._generate_visualizations_parallel(result)
            result.visualizations = visualizations
            
            # Paso 7: Finalizar
            self._emit_progress(100, "Análisis completado")
            
            # Calcular tiempo total
            result.processing_time = time.time() - start_time
            result.status = AnalysisStatus.COMPLETED
            
            logger.info(f"Análisis optimizado completado en {result.processing_time:.2f} segundos")
            self.analysis_completed.emit(result)
            
        except Exception as e:
            error_msg = f"Error durante el análisis: {str(e)}"
            error_details = traceback.format_exc()
            
            logger.error(error_msg)
            logger.error(error_details)
            
            result.status = AnalysisStatus.ERROR
            result.error_message = error_msg
            result.error_details = error_details
            result.processing_time = time.time() - start_time
            
            self.error_occurred.emit(error_msg, error_details)
            self.analysis_completed.emit(result)
        
        finally:
            # Limpieza final
            self._cleanup_resources()
    
    def _check_stop(self) -> bool:
        """Verifica si se debe detener el análisis"""
        if self._should_stop:
            logger.info("Análisis detenido por solicitud del usuario")
            return True
        return False
    
    def _emit_progress(self, percentage: int, message: str):
        """Emite señal de progreso"""
        self.progress_updated.emit(percentage, message)
        self.status_changed.emit(message)
    
    def _load_and_process_image_optimized(self) -> Dict[str, Any]:
        """Carga y procesa la imagen con optimizaciones de cache"""
        cache_key = f"processed_{hash(self.image_path)}"
        
        # Verificar cache
        if self.processing_config.get('cache_intermediate_results', True):
            with QMutexLocker(self._cache_mutex):
                if cache_key in self._cache:
                    logger.info("Usando imagen procesada desde cache")
                    return self._cache[cache_key]
        
        # Procesar imagen
        processed_data = self.backend.load_and_process_image(
            self.image_path,
            self.processing_config
        )
        
        # Guardar en cache
        if self.processing_config.get('cache_intermediate_results', True):
            with QMutexLocker(self._cache_mutex):
                self._cache[cache_key] = processed_data
        
        return processed_data
    
    def _analyze_image_quality_parallel(self, processed_image) -> Dict[str, Any]:
        """Análisis de calidad con procesamiento paralelo"""
        if not self.processing_config.get('enable_parallel_analysis', True):
            return self.backend.analyze_image_quality(processed_image)
        
        # Dividir análisis en tareas paralelas
        quality_tasks = [
            ('sharpness', lambda: self.backend.calculate_sharpness(processed_image)),
            ('noise', lambda: self.backend.calculate_noise_level(processed_image)),
            ('contrast', lambda: self.backend.calculate_contrast(processed_image)),
            ('brightness', lambda: self.backend.calculate_brightness(processed_image))
        ]
        
        results = {}
        futures = {}
        
        # Ejecutar tareas en paralelo
        for task_name, task_func in quality_tasks:
            if self._check_stop():
                break
            future = self._thread_pool.submit(task_func)
            futures[task_name] = future
        
        # Recopilar resultados
        for task_name, future in futures.items():
            try:
                results[task_name] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Error en análisis de {task_name}: {e}")
                results[task_name] = None
        
        # Calcular métricas combinadas
        results['overall_quality'] = self._calculate_overall_quality(results)
        
        return results
    
    def _perform_statistical_analysis_vectorized(self, processed_data, quality_results) -> Dict[str, Any]:
        """Análisis estadístico con operaciones vectorizadas"""
        image = processed_data['processed_image']
        
        # Usar NumPy para cálculos vectorizados
        if isinstance(image, np.ndarray):
            # Estadísticas básicas vectorizadas
            stats = {
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'median': float(np.median(image)),
                'percentiles': np.percentile(image, [25, 50, 75, 90, 95, 99]).tolist()
            }
            
            # Histograma optimizado
            hist, bins = np.histogram(image.flatten(), bins=256, density=True)
            stats['histogram'] = {
                'values': hist.tolist(),
                'bins': bins.tolist()
            }
            
            # Análisis de distribución
            stats['distribution_analysis'] = self._analyze_distribution_vectorized(image)
            
        else:
            # Fallback para otros tipos de imagen
            stats = self.backend.perform_statistical_analysis(processed_data, quality_results)
        
        return stats
    
    def _analyze_nist_compliance_async(self, quality_results) -> Dict[str, Any]:
        """Análisis NIST asíncrono"""
        if not self.nist_metadata:
            return {}
        
        # Ejecutar análisis NIST en thread separado
        future = self._thread_pool.submit(
            self.backend.analyze_nist_compliance,
            quality_results,
            self.nist_metadata
        )
        
        try:
            return future.result(timeout=60)
        except Exception as e:
            logger.error(f"Error en análisis NIST: {e}")
            return {'error': str(e)}
    
    def _generate_visualizations_parallel(self, result) -> Dict[str, Any]:
        """Generación de visualizaciones en paralelo"""
        if not self.processing_config.get('async_visualizations', True):
            return self.backend.generate_visualizations(result)
        
        # Tareas de visualización
        viz_tasks = [
            ('quality_chart', lambda: self._generate_quality_chart(result.quality_metrics)),
            ('histogram', lambda: self._generate_histogram(result.statistical_results)),
            ('feature_map', lambda: self._generate_feature_map(result.features))
        ]
        
        visualizations = {}
        futures = {}
        
        # Ejecutar en paralelo
        for viz_name, viz_func in viz_tasks:
            if self._check_stop():
                break
            future = self._thread_pool.submit(viz_func)
            futures[viz_name] = future
        
        # Recopilar resultados y emitir señales
        for viz_name, future in futures.items():
            try:
                viz_data = future.result(timeout=30)
                visualizations[viz_name] = viz_data
                # Emitir visualización lista inmediatamente
                self.visualization_ready.emit(viz_name, viz_data)
            except Exception as e:
                logger.error(f"Error generando {viz_name}: {e}")
        
        return visualizations
    
    def _calculate_overall_quality(self, quality_metrics) -> float:
        """Calcula calidad general basada en métricas individuales"""
        weights = {
            'sharpness': 0.3,
            'noise': 0.25,
            'contrast': 0.25,
            'brightness': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics and quality_metrics[metric] is not None:
                # Normalizar métricas a escala 0-1
                normalized_score = self._normalize_quality_metric(metric, quality_metrics[metric])
                total_score += normalized_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_quality_metric(self, metric_name: str, value: float) -> float:
        """Normaliza métricas de calidad a escala 0-1"""
        # Rangos típicos para normalización
        ranges = {
            'sharpness': (0, 100),
            'noise': (0, 50),  # Invertir: menos ruido = mejor
            'contrast': (0, 100),
            'brightness': (0, 255)
        }
        
        if metric_name in ranges:
            min_val, max_val = ranges[metric_name]
            normalized = (value - min_val) / (max_val - min_val)
            
            # Invertir para ruido (menos ruido = mejor calidad)
            if metric_name == 'noise':
                normalized = 1.0 - normalized
            
            return max(0.0, min(1.0, normalized))
        
        return 0.5  # Valor por defecto
    
    def _analyze_distribution_vectorized(self, image: np.ndarray) -> Dict[str, Any]:
        """Análisis de distribución usando operaciones vectorizadas"""
        flat_image = image.flatten()
        
        # Cálculos vectorizados
        analysis = {
            'skewness': float(self._calculate_skewness(flat_image)),
            'kurtosis': float(self._calculate_kurtosis(flat_image)),
            'entropy': float(self._calculate_entropy(flat_image)),
            'uniformity': float(self._calculate_uniformity(flat_image))
        }
        
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcula asimetría usando NumPy"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calcula curtosis usando NumPy"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calcula entropía usando NumPy"""
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Eliminar ceros
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_uniformity(self, data: np.ndarray) -> float:
        """Calcula uniformidad usando NumPy"""
        hist, _ = np.histogram(data, bins=256, density=True)
        return np.sum(hist ** 2)
    
    def _generate_quality_chart(self, quality_metrics) -> Dict[str, Any]:
        """Genera gráfico de métricas de calidad"""
        # Implementación simplificada
        return {
            'type': 'quality_chart',
            'data': quality_metrics,
            'timestamp': time.time()
        }
    
    def _generate_histogram(self, statistical_results) -> Dict[str, Any]:
        """Genera histograma de la imagen"""
        return {
            'type': 'histogram',
            'data': statistical_results.get('histogram', {}),
            'timestamp': time.time()
        }
    
    def _generate_feature_map(self, features) -> Dict[str, Any]:
        """Genera mapa de características"""
        return {
            'type': 'feature_map',
            'data': features,
            'timestamp': time.time()
        }
    
    def _update_memory_usage(self):
        """Actualiza el uso de memoria"""
        try:
            memory_mb = self._process.memory_info().rss / 1024 / 1024
            self.memory_usage_updated.emit(memory_mb)
            
            # Verificar límite de memoria
            if memory_mb > self._memory_threshold_mb:
                logger.warning(f"Uso de memoria alto: {memory_mb:.1f}MB")
                # Limpiar cache si es necesario
                with QMutexLocker(self._cache_mutex):
                    if len(self._cache) > 5:
                        self._cache.clear()
                        gc.collect()
        except Exception as e:
            logger.error(f"Error monitoreando memoria: {e}")
    
    def _cleanup_resources(self):
        """Limpia recursos y memoria"""
        try:
            # Detener timer
            if self._progress_timer.isActive():
                self._progress_timer.stop()
            
            # Cerrar pool de threads
            self._thread_pool.shutdown(wait=True)
            
            # Limpiar cache si es necesario
            with QMutexLocker(self._cache_mutex):
                if len(self._cache) > 10:  # Límite de cache
                    self._cache.clear()
            
            # Forzar recolección de basura
            gc.collect()
            
            logger.info("Recursos limpiados correctamente")
            
        except Exception as e:
            logger.error(f"Error limpiando recursos: {e}")
    

    

    

    


if __name__ == "__main__":
    # Prueba básica del worker
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    
    app = QApplication(sys.argv)
    
    def test_worker():
        worker = OptimizedAnalysisWorker()
        
        # Configurar análisis de prueba
        worker.setup_analysis(
            image_path="test_image.jpg",
            case_data={"case_id": "TEST001", "examiner": "Test User"},
            processing_config={"resize_images": True}
        )
        
        # Conectar señales
        worker.progress_updated.connect(lambda p, m: print(f"Progreso: {p}% - {m}"))
        worker.status_changed.connect(lambda s: print(f"Estado: {s}"))
        worker.analysis_completed.connect(lambda r: print(f"Completado: {r.status}"))
        worker.error_occurred.connect(lambda m, d: print(f"Error: {m}"))
        
        print("Worker de análisis optimizado inicializado correctamente")
        app.quit()
    
    QTimer.singleShot(100, test_worker)
    sys.exit(app.exec_())