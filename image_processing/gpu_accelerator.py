"""
Aceleración GPU para Algoritmos de Procesamiento de Imágenes
Sistema Balístico Forense SIGeC-Balistica

Este módulo implementa aceleración GPU para operaciones intensivas de procesamiento
de imágenes utilizando OpenCV GPU (cv2.cuda) y CuPy para operaciones NumPy.

Características:
- Detección automática de GPU
- Fallback automático a CPU
- Aceleración de operaciones OpenCV
- Aceleración de operaciones NumPy con CuPy
- Gestión de memoria GPU
- Benchmarking de rendimiento

Autor: Sistema SIGeC-Balistica
Fecha: 2024
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field
import time
import warnings
import threading
from contextlib import contextmanager
import gc
import weakref

# Intentar importar CuPy para aceleración NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Configurar logging
logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Información sobre la GPU disponible"""
    gpu_available: bool
    opencv_gpu_available: bool
    cupy_available: bool
    gpu_count: int
    gpu_memory_mb: Optional[int] = None
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None

@dataclass
class GPUMemoryStats:
    """Estadísticas de memoria GPU"""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    allocated_blocks: int = 0
    peak_usage_bytes: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0

class GPUMemoryManager:
    """
    Gestor de memoria GPU con context managers y liberación automática
    """
    
    def __init__(self, gpu_accelerator: 'GPUAccelerator'):
        self.gpu_accelerator = gpu_accelerator
        self.active_contexts: List[weakref.ReferenceType] = []
        self.memory_threshold = 0.8  # 80% de memoria antes de limpiar
        self.auto_cleanup = True
        self._lock = threading.Lock()
        self._peak_usage = 0
        self._allocation_count = 0
        
    def __enter__(self):
        """Entrada del context manager"""
        if self.gpu_accelerator.is_gpu_enabled():
            self._initial_memory = self._get_memory_usage()
            logger.debug(f"Iniciando context GPU - Memoria inicial: {self._initial_memory} bytes")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Salida del context manager con limpieza automática"""
        if self.gpu_accelerator.is_gpu_enabled():
            try:
                self._cleanup_context_memory()
                final_memory = self._get_memory_usage()
                logger.debug(f"Finalizando context GPU - Memoria final: {final_memory} bytes")
                
                # Forzar limpieza si se excede el umbral
                if self._should_force_cleanup():
                    self.force_cleanup()
                    
            except Exception as e:
                logger.warning(f"Error en limpieza de context GPU: {e}")
                
    def _get_memory_usage(self) -> int:
        """Obtener uso actual de memoria GPU"""
        if not self.gpu_accelerator.is_gpu_enabled() or not CUPY_AVAILABLE:
            return 0
            
        try:
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes()
        except Exception:
            return 0
            
    def _should_force_cleanup(self) -> bool:
        """Determinar si se debe forzar la limpieza de memoria"""
        if not self.auto_cleanup or not CUPY_AVAILABLE:
            return False
            
        try:
            mempool = cp.get_default_memory_pool()
            total_memory = self.gpu_accelerator.gpu_info.gpu_memory_mb * 1024 * 1024
            if total_memory and mempool.used_bytes() > (total_memory * self.memory_threshold):
                return True
        except Exception:
            pass
            
        return False
        
    def _cleanup_context_memory(self):
        """Limpiar memoria específica del contexto"""
        if CUPY_AVAILABLE:
            try:
                # Forzar garbage collection
                gc.collect()
                
                # Limpiar arrays temporales de CuPy
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                
            except Exception as e:
                logger.warning(f"Error en limpieza de contexto: {e}")
                
    def force_cleanup(self):
        """Forzar limpieza completa de memoria GPU"""
        with self._lock:
            if self.gpu_accelerator.is_gpu_enabled():
                try:
                    logger.info("Ejecutando limpieza forzada de memoria GPU")
                    
                    # Limpiar memoria CuPy
                    if CUPY_AVAILABLE:
                        mempool = cp.get_default_memory_pool()
                        mempool.free_all_blocks()
                        
                    # Garbage collection agresivo
                    gc.collect()
                    
                    # Sincronizar GPU
                    if hasattr(cp, 'cuda') and CUPY_AVAILABLE:
                        cp.cuda.Stream.null.synchronize()
                        
                    logger.info("Limpieza forzada completada")
                    
                except Exception as e:
                    logger.error(f"Error en limpieza forzada: {e}")
                    
    def get_memory_stats(self) -> GPUMemoryStats:
        """Obtener estadísticas detalladas de memoria"""
        stats = GPUMemoryStats()
        
        if not self.gpu_accelerator.is_gpu_enabled():
            return stats
            
        try:
            if CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                stats.used_bytes = mempool.used_bytes()
                stats.total_bytes = self.gpu_accelerator.gpu_info.gpu_memory_mb * 1024 * 1024 if self.gpu_accelerator.gpu_info.gpu_memory_mb else 0
                stats.free_bytes = stats.total_bytes - stats.used_bytes if stats.total_bytes > 0 else 0
                stats.peak_usage_bytes = max(self._peak_usage, stats.used_bytes)
                stats.allocation_count = self._allocation_count
                
                # Actualizar pico de uso
                self._peak_usage = stats.peak_usage_bytes
                
        except Exception as e:
            logger.warning(f"Error obteniendo estadísticas de memoria: {e}")
            
        return stats
        
    def set_memory_threshold(self, threshold: float):
        """Configurar umbral de memoria para limpieza automática"""
        if 0.1 <= threshold <= 0.95:
            self.memory_threshold = threshold
            logger.info(f"Umbral de memoria configurado a {threshold*100}%")
        else:
            logger.warning("Umbral debe estar entre 0.1 y 0.95")
            
    @contextmanager
    def batch_context(self, expected_operations: int = 1):
        """Context manager optimizado para operaciones batch"""
        if not self.gpu_accelerator.is_gpu_enabled():
            yield
            return
            
        initial_memory = self._get_memory_usage()
        logger.debug(f"Iniciando batch context - {expected_operations} operaciones esperadas")
        
        try:
            # Pre-limpiar si es necesario
            if self._should_force_cleanup():
                self.force_cleanup()
                
            yield
            
        finally:
            # Limpieza post-batch
            final_memory = self._get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            logger.debug(f"Batch completado - Memoria utilizada: {memory_diff} bytes")
            
            # Limpiar si se usó mucha memoria
            if memory_diff > (50 * 1024 * 1024):  # 50MB
                self._cleanup_context_memory()

class GPUAccelerator:
    """
    Acelerador GPU para operaciones de procesamiento de imágenes
    """
    
    def __init__(self, enable_gpu: bool = True, device_id: int = 0):
        """
        Inicializar el acelerador GPU
        
        Args:
            enable_gpu: Habilitar aceleración GPU
            device_id: ID del dispositivo GPU a utilizar
        """
        self.enable_gpu = enable_gpu
        self.device_id = device_id
        self.gpu_info = self._detect_gpu_capabilities()
        
        # Inicializar gestor de memoria
        self.memory_manager = GPUMemoryManager(self)
        
        # Configurar dispositivo GPU si está disponible
        if self.gpu_info.gpu_available and enable_gpu:
            try:
                cv2.cuda.setDevice(device_id)
                if CUPY_AVAILABLE:
                    cp.cuda.Device(device_id).use()
                logger.info(f"GPU {device_id} configurada correctamente")
            except Exception as e:
                logger.warning(f"Error configurando GPU {device_id}: {e}")
                self.enable_gpu = False
        
        # Crear streams para operaciones asíncronas
        self.stream = cv2.cuda_Stream() if self.gpu_info.opencv_gpu_available and enable_gpu else None
        
        logger.info(f"GPUAccelerator inicializado - GPU habilitada: {self.is_gpu_enabled()}")
    
    def _detect_gpu_capabilities(self) -> GPUInfo:
        """Detectar capacidades de GPU disponibles"""
        gpu_available = False
        opencv_gpu_available = False
        gpu_count = 0
        gpu_memory_mb = None
        gpu_name = None
        compute_capability = None
        
        try:
            # Verificar OpenCV GPU
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                gpu_available = True
                opencv_gpu_available = True
                
                # Obtener información del dispositivo
                device_info = cv2.cuda.DeviceInfo(self.device_id)
                gpu_name = device_info.name()
                gpu_memory_mb = device_info.totalMemory() // (1024 * 1024)
                compute_capability = f"{device_info.majorVersion()}.{device_info.minorVersion()}"
                
                logger.info(f"GPU detectada: {gpu_name} ({gpu_memory_mb}MB, CC {compute_capability})")
            else:
                logger.info("No se detectaron GPUs compatibles con OpenCV")
                
        except Exception as e:
            logger.warning(f"Error detectando capacidades GPU: {e}")
        
        return GPUInfo(
            gpu_available=gpu_available,
            opencv_gpu_available=opencv_gpu_available,
            cupy_available=CUPY_AVAILABLE,
            gpu_count=gpu_count,
            gpu_memory_mb=gpu_memory_mb,
            gpu_name=gpu_name,
            compute_capability=compute_capability
        )
    
    def is_gpu_enabled(self) -> bool:
        """Verificar si la GPU está habilitada y disponible"""
        return self.enable_gpu and self.gpu_info.gpu_available
    
    def get_gpu_info(self) -> GPUInfo:
        """Obtener información de la GPU"""
        return self.gpu_info
    
    # ========================================================================
    # OPERACIONES BÁSICAS DE IMAGEN
    # ========================================================================
    
    def resize(self, image: np.ndarray, size: Tuple[int, int], 
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Redimensionar imagen con aceleración GPU
        
        Args:
            image: Imagen de entrada
            size: Nuevo tamaño (width, height)
            interpolation: Método de interpolación
            
        Returns:
            Imagen redimensionada
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                # Subir imagen a GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Redimensionar en GPU
                gpu_result = cv2.cuda.resize(gpu_img, size, interpolation=interpolation, stream=self.stream)
                
                # Descargar resultado
                result = gpu_result.download()
                return result
                
            except Exception as e:
                logger.warning(f"Error en resize GPU, usando CPU: {e}")
        
        # Fallback a CPU
        return cv2.resize(image, size, interpolation=interpolation)
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int], 
                     sigma_x: float, sigma_y: float = 0) -> np.ndarray:
        """
        Filtro Gaussiano con aceleración GPU
        
        Args:
            image: Imagen de entrada
            kernel_size: Tamaño del kernel
            sigma_x: Desviación estándar en X
            sigma_y: Desviación estándar en Y
            
        Returns:
            Imagen filtrada
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                gpu_result = cv2.cuda.GaussianBlur(gpu_img, kernel_size, sigma_x, sigmaY=sigma_y, stream=self.stream)
                
                result = gpu_result.download()
                return result
                
            except Exception as e:
                logger.warning(f"Error en GaussianBlur GPU, usando CPU: {e}")
        
        # Fallback a CPU
        return cv2.GaussianBlur(image, kernel_size, sigma_x, sigmaY=sigma_y)
    
    def bilateral_filter(self, image: np.ndarray, d: int, sigma_color: float, 
                        sigma_space: float) -> np.ndarray:
        """
        Filtro bilateral con aceleración GPU
        
        Args:
            image: Imagen de entrada
            d: Diámetro del vecindario
            sigma_color: Sigma para el espacio de color
            sigma_space: Sigma para el espacio de coordenadas
            
        Returns:
            Imagen filtrada
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                gpu_result = cv2.cuda.bilateralFilter(gpu_img, d, sigma_color, sigma_space, stream=self.stream)
                
                result = gpu_result.download()
                return result
                
            except Exception as e:
                logger.warning(f"Error en bilateralFilter GPU, usando CPU: {e}")
        
        # Fallback a CPU
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def morphology_ex(self, image: np.ndarray, op: int, kernel: np.ndarray, 
                     iterations: int = 1) -> np.ndarray:
        """
        Operaciones morfológicas con aceleración GPU
        
        Args:
            image: Imagen de entrada
            op: Tipo de operación morfológica
            kernel: Elemento estructurante
            iterations: Número de iteraciones
            
        Returns:
            Imagen procesada
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                gpu_result = cv2.cuda.morphologyEx(gpu_img, op, kernel, iterations=iterations, stream=self.stream)
                
                result = gpu_result.download()
                return result
                
            except Exception as e:
                logger.warning(f"Error en morphologyEx GPU, usando CPU: {e}")
        
        # Fallback a CPU
        return cv2.morphologyEx(image, op, kernel, iterations=iterations)
    
    def threshold(self, image: np.ndarray, thresh: float, maxval: float, 
                 type: int) -> Tuple[float, np.ndarray]:
        """
        Umbralización con aceleración GPU
        
        Args:
            image: Imagen de entrada
            thresh: Valor de umbral
            maxval: Valor máximo asignado
            type: Tipo de umbralización
            
        Returns:
            Tupla (threshold_value, thresholded_image)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                ret, gpu_result = cv2.cuda.threshold(gpu_img, thresh, maxval, type, stream=self.stream)
                
                result = gpu_result.download()
                return ret, result
                
            except Exception as e:
                logger.warning(f"Error en threshold GPU, usando CPU: {e}")
        
        # Fallback a CPU
        return cv2.threshold(image, thresh, maxval, type)
    
    def clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) con aceleración GPU
        
        Args:
            image: Imagen de entrada en escala de grises
            clip_limit: Límite de recorte
            tile_grid_size: Tamaño de la grilla de tiles
            
        Returns:
            Imagen con contraste mejorado
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                clahe_gpu = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                gpu_result = clahe_gpu.apply(gpu_img, stream=self.stream)
                
                result = gpu_result.download()
                return result
                
            except Exception as e:
                logger.warning(f"Error en CLAHE GPU, usando CPU: {e}")
        
        # Fallback a CPU
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    # ========================================================================
    # ACELERACIÓN DE ALGORITMOS DE EXTRACCIÓN DE CARACTERÍSTICAS
    # ========================================================================
    
    def extract_orb_features(self, image: np.ndarray, max_features: int = 1000) -> Tuple[list, np.ndarray]:
        """
        Extracción de características ORB con aceleración GPU
        
        Args:
            image: Imagen de entrada
            max_features: Número máximo de características
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                orb_gpu = cv2.cuda_ORB.create(nfeatures=max_features)
                keypoints_gpu, descriptors_gpu = orb_gpu.detectAndComputeAsync(gpu_img, None, stream=self.stream)
                
                # Descargar resultados
                if descriptors_gpu is not None:
                    descriptors = descriptors_gpu.download()
                else:
                    descriptors = None
                
                return keypoints_gpu, descriptors
                
            except Exception as e:
                logger.warning(f"Error en ORB GPU, usando CPU: {e}")
        
        # Fallback a CPU
        orb = cv2.ORB_create(nfeatures=max_features)
        return orb.detectAndCompute(image, None)
    
    def extract_sift_features(self, image: np.ndarray, max_features: int = 1000) -> Tuple[list, np.ndarray]:
        """
        Extracción de características SIFT con aceleración GPU
        
        Args:
            image: Imagen de entrada
            max_features: Número máximo de características
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                # Nota: SIFT GPU requiere OpenCV compilado con soporte CUDA específico
                # Si no está disponible, usar CPU
                sift_gpu = cv2.cuda_SIFT.create(nfeatures=max_features)
                keypoints_gpu, descriptors_gpu = sift_gpu.detectAndComputeAsync(gpu_img, None, stream=self.stream)
                
                # Descargar resultados
                if descriptors_gpu is not None:
                    descriptors = descriptors_gpu.download()
                else:
                    descriptors = None
                
                return keypoints_gpu, descriptors
                
            except Exception as e:
                logger.warning(f"SIFT GPU no disponible, usando CPU: {e}")
        
        # Fallback a CPU
        sift = cv2.SIFT_create(nfeatures=max_features)
        return sift.detectAndCompute(image, None)
    
    def extract_akaze_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Extracción de características AKAZE con aceleración GPU
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                akaze_gpu = cv2.cuda_AKAZE.create()
                keypoints_gpu, descriptors_gpu = akaze_gpu.detectAndComputeAsync(gpu_img, None, stream=self.stream)
                
                # Descargar resultados
                if descriptors_gpu is not None:
                    descriptors = descriptors_gpu.download()
                else:
                    descriptors = None
                
                return keypoints_gpu, descriptors
                
            except Exception as e:
                logger.warning(f"AKAZE GPU no disponible, usando CPU: {e}")
        
        # Fallback a CPU
        akaze = cv2.AKAZE_create()
        return akaze.detectAndCompute(image, None)
    
    def extract_brisk_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Extracción de características BRISK con aceleración GPU
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                brisk_gpu = cv2.cuda_BRISK.create()
                keypoints_gpu, descriptors_gpu = brisk_gpu.detectAndComputeAsync(gpu_img, None, stream=self.stream)
                
                # Descargar resultados
                if descriptors_gpu is not None:
                    descriptors = descriptors_gpu.download()
                else:
                    descriptors = None
                
                return keypoints_gpu, descriptors
                
            except Exception as e:
                logger.warning(f"BRISK GPU no disponible, usando CPU: {e}")
        
        # Fallback a CPU
        brisk = cv2.BRISK_create()
        return brisk.detectAndCompute(image, None)
    
    def extract_kaze_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Extracción de características KAZE con aceleración GPU
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                
                kaze_gpu = cv2.cuda_KAZE.create()
                keypoints_gpu, descriptors_gpu = kaze_gpu.detectAndComputeAsync(gpu_img, None, stream=self.stream)
                
                # Descargar resultados
                if descriptors_gpu is not None:
                    descriptors = descriptors_gpu.download()
                else:
                    descriptors = None
                
                return keypoints_gpu, descriptors
                
            except Exception as e:
                logger.warning(f"KAZE GPU no disponible, usando CPU: {e}")
        
        # Fallback a CPU
        kaze = cv2.KAZE_create()
        return kaze.detectAndCompute(image, None)
    
    def match_descriptors_gpu(self, desc1: np.ndarray, desc2: np.ndarray, 
                             matcher_type: str = "BF", norm_type: int = cv2.NORM_HAMMING,
                             max_memory_mb: int = 1000) -> list:
        """
        Matching de descriptores con aceleración GPU y gestión de memoria optimizada
        
        Args:
            desc1: Descriptores de la primera imagen
            desc2: Descriptores de la segunda imagen
            matcher_type: Tipo de matcher ("BF" para BruteForce, "FLANN" para FLANN)
            norm_type: Tipo de norma para el matching
            max_memory_mb: Límite máximo de memoria GPU en MB
            
        Returns:
            Lista de matches
        """
        if self.is_gpu_enabled() and self.gpu_info.opencv_gpu_available:
            try:
                # Verificar memoria disponible antes de procesar
                memory_info = self.get_memory_info()
                if memory_info.get('gpu', {}).get('used_mb', 0) > max_memory_mb:
                    logger.warning(f"Memoria GPU insuficiente, usando CPU")
                    return self._match_descriptors_cpu(desc1, desc2, matcher_type, norm_type)
                
                # Usar context manager para gestión automática de memoria
                with self.gpu_memory_context():
                    gpu_desc1 = cv2.cuda_GpuMat()
                    gpu_desc2 = cv2.cuda_GpuMat()
                    
                    # Subir descriptores a GPU
                    gpu_desc1.upload(desc1)
                    gpu_desc2.upload(desc2)
                    
                    if matcher_type == "BF":
                        matcher_gpu = cv2.cuda_DescriptorMatcher.createBFMatcher(norm_type)
                    else:
                        # FLANN GPU matcher
                        matcher_gpu = cv2.cuda_DescriptorMatcher.createBFMatcher(norm_type)
                    
                    matches_gpu = matcher_gpu.match(gpu_desc1, gpu_desc2)
                    
                    # Limpiar memoria GPU explícitamente
                    gpu_desc1.release()
                    gpu_desc2.release()
                    
                    return matches_gpu
                
            except Exception as e:
                logger.warning(f"Matching GPU no disponible, usando CPU: {e}")
                return self._match_descriptors_cpu(desc1, desc2, matcher_type, norm_type)
        
        # Fallback a CPU
        return self._match_descriptors_cpu(desc1, desc2, matcher_type, norm_type)
    
    def _match_descriptors_cpu(self, desc1: np.ndarray, desc2: np.ndarray, 
                              matcher_type: str, norm_type: int) -> list:
        """Método auxiliar para matching en CPU"""
        if matcher_type == "BF":
            matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        else:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        return matcher.match(desc1, desc2)
    
    # ========================================================================
    # OPERACIONES NUMPY CON CUPY
    # ========================================================================
    
    # ========================================================================
    # ACELERACIÓN DE OPERACIONES NUMPY CON CUPY
    # ========================================================================
    
    def array_multiply(self, a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
        """
        Multiplicación de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Primer array
            b: Segundo array o escalar
            
        Returns:
            Resultado de la multiplicación
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy arrays
                a_gpu = cp.asarray(a)
                if isinstance(b, np.ndarray):
                    b_gpu = cp.asarray(b)
                else:
                    b_gpu = b
                
                # Realizar operación en GPU
                result_gpu = cp.multiply(a_gpu, b_gpu)
                
                # Convertir de vuelta a NumPy
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en multiplicación GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.multiply(a, b)
    
    def array_add(self, a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
        """
        Suma de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Primer array
            b: Segundo array o escalar
            
        Returns:
            Resultado de la suma
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy arrays
                a_gpu = cp.asarray(a)
                if isinstance(b, np.ndarray):
                    b_gpu = cp.asarray(b)
                else:
                    b_gpu = b
                
                # Realizar operación en GPU
                result_gpu = cp.add(a_gpu, b_gpu)
                
                # Convertir de vuelta a NumPy
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en suma GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.add(a, b)
    
    def array_subtract(self, a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
        """
        Resta de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Primer array
            b: Segundo array o escalar
            
        Returns:
            Resultado de la resta
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy arrays
                a_gpu = cp.asarray(a)
                if isinstance(b, np.ndarray):
                    b_gpu = cp.asarray(b)
                else:
                    b_gpu = b
                
                # Realizar operación en GPU
                result_gpu = cp.subtract(a_gpu, b_gpu)
                
                # Convertir de vuelta a NumPy
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en resta GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.subtract(a, b)
    
    def array_sqrt(self, a: np.ndarray) -> np.ndarray:
        """
        Raíz cuadrada de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Array de entrada
            
        Returns:
            Raíz cuadrada del array
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy array
                a_gpu = cp.asarray(a)
                
                # Realizar operación en GPU
                result_gpu = cp.sqrt(a_gpu)
                
                # Convertir de vuelta a NumPy
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en sqrt GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.sqrt(a)
    
    def array_mean(self, a: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> Union[np.ndarray, float]:
        """
        Media de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Array de entrada
            axis: Eje(s) sobre los que calcular la media
            
        Returns:
            Media del array
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy array
                a_gpu = cp.asarray(a)
                
                # Realizar operación en GPU
                result_gpu = cp.mean(a_gpu, axis=axis)
                
                # Convertir de vuelta a NumPy
                if isinstance(result_gpu, cp.ndarray):
                    return cp.asnumpy(result_gpu)
                else:
                    return float(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en mean GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.mean(a, axis=axis)
    
    def array_std(self, a: np.ndarray, axis: Optional[Union[int, tuple]] = None) -> Union[np.ndarray, float]:
        """
        Desviación estándar de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Array de entrada
            axis: Eje(s) sobre los que calcular la desviación estándar
            
        Returns:
            Desviación estándar del array
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy array
                a_gpu = cp.asarray(a)
                
                # Realizar operación en GPU
                result_gpu = cp.std(a_gpu, axis=axis)
                
                # Convertir de vuelta a NumPy
                if isinstance(result_gpu, cp.ndarray):
                    return cp.asnumpy(result_gpu)
                else:
                    return float(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en std GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.std(a, axis=axis)
    
    def array_dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Producto punto de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Primer array
            b: Segundo array
            
        Returns:
            Producto punto de los arrays
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy arrays
                a_gpu = cp.asarray(a)
                b_gpu = cp.asarray(b)
                
                # Realizar operación en GPU
                result_gpu = cp.dot(a_gpu, b_gpu)
                
                # Convertir de vuelta a NumPy
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en dot GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.dot(a, b)
    
    def array_norm(self, a: np.ndarray, ord: Optional[Union[int, float, str]] = None, 
                   axis: Optional[Union[int, tuple]] = None) -> Union[np.ndarray, float]:
        """
        Norma de arrays con aceleración GPU usando CuPy
        
        Args:
            a: Array de entrada
            ord: Orden de la norma
            axis: Eje(s) sobre los que calcular la norma
            
        Returns:
            Norma del array
        """
        if self.is_gpu_enabled() and self.gpu_info.cupy_available:
            try:
                # Convertir a CuPy array
                a_gpu = cp.asarray(a)
                
                # Realizar operación en GPU
                result_gpu = cp.linalg.norm(a_gpu, ord=ord, axis=axis)
                
                # Convertir de vuelta a NumPy
                if isinstance(result_gpu, cp.ndarray):
                    return cp.asnumpy(result_gpu)
                else:
                    return float(result_gpu)
                
            except Exception as e:
                logger.warning(f"Error en norm GPU, usando CPU: {e}")
        
        # Fallback a NumPy CPU
        return np.linalg.norm(a, ord=ord, axis=axis)
    
    # ========================================================================
    # BENCHMARKING Y UTILIDADES
    # ========================================================================
    
    def benchmark_operation(self, operation_name: str, operation_func, 
                           *args, iterations: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Benchmark de una operación
        
        Args:
            operation_name: Nombre de la operación
            operation_func: Función a benchmarear
            *args: Argumentos posicionales
            iterations: Número de iteraciones
            **kwargs: Argumentos con nombre
            
        Returns:
            Diccionario con resultados del benchmark
        """
        times = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            result = operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'operation': operation_name,
            'iterations': iterations,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times),
            'gpu_enabled': self.is_gpu_enabled()
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Obtener información de memoria GPU
        
        Returns:
            Diccionario con información de memoria
        """
        if not self.is_gpu_enabled():
            return {'gpu_enabled': False}
        
        try:
            if CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                return {
                    'gpu_enabled': True,
                    'used_bytes': mempool.used_bytes(),
                    'total_bytes': mempool.total_bytes(),
                    'free_bytes': self.gpu_info.gpu_memory_mb * 1024 * 1024 - mempool.used_bytes() if self.gpu_info.gpu_memory_mb else None
                }
            else:
                return {
                    'gpu_enabled': True,
                    'cupy_available': False,
                    'total_memory_mb': self.gpu_info.gpu_memory_mb
                }
        except Exception as e:
            logger.warning(f"Error obteniendo información de memoria: {e}")
            return {'gpu_enabled': True, 'error': str(e)}
    
    def clear_gpu_memory(self):
        """Limpiar memoria GPU usando el gestor de memoria mejorado"""
        if self.is_gpu_enabled():
            self.memory_manager.force_cleanup()
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager para gestión automática de memoria GPU"""
        with self.memory_manager:
            yield
            
    @contextmanager 
    def gpu_batch_context(self, expected_operations: int = 1):
        """Context manager optimizado para operaciones batch"""
        with self.memory_manager.batch_context(expected_operations):
            yield
            
    def get_detailed_memory_info(self) -> Dict[str, Any]:
        """
        Obtener información detallada de memoria GPU incluyendo estadísticas
        
        Returns:
            Diccionario con información completa de memoria
        """
        if not self.is_gpu_enabled():
            return {'gpu_enabled': False}
        
        try:
            # Obtener estadísticas del gestor de memoria
            stats = self.memory_manager.get_memory_stats()
            
            # Información básica existente
            basic_info = self.get_memory_info()
            
            # Combinar información
            detailed_info = {
                **basic_info,
                'detailed_stats': {
                    'total_bytes': stats.total_bytes,
                    'used_bytes': stats.used_bytes,
                    'free_bytes': stats.free_bytes,
                    'peak_usage_bytes': stats.peak_usage_bytes,
                    'allocation_count': stats.allocation_count,
                    'memory_threshold': self.memory_manager.memory_threshold,
                    'auto_cleanup_enabled': self.memory_manager.auto_cleanup
                }
            }
            
            return detailed_info
            
        except Exception as e:
            logger.warning(f"Error obteniendo información detallada de memoria: {e}")
            return self.get_memory_info()
    
    def set_memory_management_config(self, threshold: float = 0.8, auto_cleanup: bool = True):
        """
        Configurar parámetros de gestión de memoria
        
        Args:
            threshold: Umbral de memoria para limpieza automática (0.1-0.95)
            auto_cleanup: Habilitar limpieza automática
        """
        self.memory_manager.set_memory_threshold(threshold)
        self.memory_manager.auto_cleanup = auto_cleanup
        logger.info(f"Configuración de memoria actualizada - Umbral: {threshold*100}%, Auto-cleanup: {auto_cleanup}")

# Instancia global del acelerador GPU
_gpu_accelerator = None

def get_gpu_accelerator(enable_gpu: bool = True, device_id: int = 0) -> GPUAccelerator:
    """
    Obtener instancia global del acelerador GPU
    
    Args:
        enable_gpu: Habilitar GPU
        device_id: ID del dispositivo
        
    Returns:
        Instancia del acelerador GPU
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(enable_gpu=enable_gpu, device_id=device_id)
        
        # Inicializar monitoreo GPU si está disponible
        try:
            from ..performance.gpu_monitor import get_gpu_monitor, start_gpu_monitoring
            monitor = get_gpu_monitor()
            if not monitor._monitoring:
                start_gpu_monitoring()
                logger.info("Monitoreo GPU iniciado automáticamente")
        except ImportError:
            logger.debug("Monitor GPU no disponible")
        except Exception as e:
            logger.warning(f"Error iniciando monitor GPU: {e}")
    
    return _gpu_accelerator

def is_gpu_available() -> bool:
    """Verificar si hay GPU disponible"""
    accelerator = get_gpu_accelerator()
    return accelerator.is_gpu_enabled()

# Funciones de conveniencia para context managers
@contextmanager
def gpu_memory_context():
    """Context manager global para gestión de memoria GPU"""
    accelerator = get_gpu_accelerator()
    with accelerator.gpu_memory_context():
        yield

@contextmanager  
def gpu_batch_context(expected_operations: int = 1):
    """Context manager global para operaciones batch"""
    accelerator = get_gpu_accelerator()
    with accelerator.gpu_batch_context(expected_operations):
        yield

def get_gpu_info() -> GPUInfo:
    """Obtener información de GPU"""
    accelerator = get_gpu_accelerator()
    return accelerator.get_gpu_info()

def get_detailed_gpu_memory_info() -> Dict[str, Any]:
    """Obtener información detallada de memoria GPU"""
    accelerator = get_gpu_accelerator()
    return accelerator.get_detailed_memory_info()

def configure_gpu_memory_management(threshold: float = 0.8, auto_cleanup: bool = True):
    """Configurar gestión de memoria GPU globalmente"""
    accelerator = get_gpu_accelerator()
    accelerator.set_memory_management_config(threshold, auto_cleanup)