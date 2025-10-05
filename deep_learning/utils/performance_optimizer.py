"""
Optimizador de rendimiento para entrenamiento de modelos de deep learning
"""

import torch
import psutil
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Clase para optimizar el rendimiento del entrenamiento de modelos
    """
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self.system_info = self._get_system_info()
        
    def _get_optimal_device(self) -> torch.device:
        """Determinar el dispositivo óptimo para entrenamiento"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"GPU disponible: {torch.cuda.get_device_name()}")
            logger.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info("Usando CPU para entrenamiento")
        
        return device
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            return {
                'memory_gb': memory_gb,
                'cpu_count': cpu_count,
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except Exception as e:
            logger.warning(f"No se pudo obtener información del sistema: {e}")
            return {
                'memory_gb': 8.0,  # Valor por defecto
                'cpu_count': 4,    # Valor por defecto
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
    
    def optimize_training_settings(self, config: Any) -> Dict[str, Any]:
        """
        Optimizar configuraciones de entrenamiento basadas en el hardware disponible
        
        Args:
            config: Configuración de entrenamiento
            
        Returns:
            Dict con configuraciones optimizadas
        """
        optimizations = {}
        
        # Optimizar batch size basado en memoria disponible
        if hasattr(config, 'data') and hasattr(config.data, 'batch_size'):
            optimal_batch_size = self._calculate_optimal_batch_size(
                config.data.batch_size,
                config.data.image_size if hasattr(config.data, 'image_size') else [224, 224]
            )
            if optimal_batch_size != config.data.batch_size:
                optimizations['batch_size'] = optimal_batch_size
                logger.info(f"Batch size optimizado: {config.data.batch_size} -> {optimal_batch_size}")
        
        # Optimizar número de workers
        if hasattr(config, 'data') and hasattr(config.data, 'num_workers'):
            optimal_workers = min(self.system_info['cpu_count'], 8)  # Máximo 8 workers
            if optimal_workers != config.data.num_workers:
                optimizations['num_workers'] = optimal_workers
                logger.info(f"Número de workers optimizado: {config.data.num_workers} -> {optimal_workers}")
        
        # Configurar pin_memory para GPU
        if self.system_info['gpu_available'] and hasattr(config, 'data'):
            if not hasattr(config.data, 'pin_memory') or not config.data.pin_memory:
                optimizations['pin_memory'] = True
                logger.info("Pin memory habilitado para GPU")
        
        return optimizations
    
    def _calculate_optimal_batch_size(self, current_batch_size: int, image_size: list) -> int:
        """Calcular el batch size óptimo basado en memoria disponible"""
        if self.system_info['gpu_available']:
            # Para GPU, usar memoria GPU disponible
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Estimación aproximada: cada imagen RGB de 224x224 usa ~2MB en GPU
                image_memory_mb = (image_size[0] * image_size[1] * 3 * 4) / (1024**2)  # 4 bytes por float32
                max_batch_size = int((gpu_memory_gb * 1024 * 0.7) / image_memory_mb)  # 70% de memoria disponible
                
                # Ajustar a potencias de 2 para mejor rendimiento
                optimal_batch_size = min(current_batch_size, max_batch_size)
                optimal_batch_size = 2 ** int(torch.log2(torch.tensor(optimal_batch_size)).item())
                
                return max(optimal_batch_size, 4)  # Mínimo batch size de 4
            except Exception as e:
                logger.warning(f"Error calculando batch size óptimo para GPU: {e}")
                return current_batch_size
        else:
            # Para CPU, usar memoria RAM disponible
            memory_gb = self.system_info['memory_gb']
            if memory_gb < 8:
                return min(current_batch_size, 16)
            elif memory_gb < 16:
                return min(current_batch_size, 32)
            else:
                return current_batch_size
    
    def setup_torch_optimizations(self):
        """Configurar optimizaciones específicas de PyTorch"""
        try:
            # Habilitar optimizaciones de PyTorch
            torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
            torch.backends.cudnn.deterministic = False
            
            # Configurar número de threads para CPU
            if not torch.cuda.is_available():
                torch.set_num_threads(self.system_info['cpu_count'])
            
            logger.info("Optimizaciones de PyTorch configuradas")
            
        except Exception as e:
            logger.warning(f"Error configurando optimizaciones de PyTorch: {e}")
    
    def get_device(self) -> torch.device:
        """Obtener el dispositivo óptimo"""
        return self.device
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema"""
        return self.system_info.copy()
    
    def log_system_info(self):
        """Registrar información del sistema en los logs"""
        info = self.system_info
        logger.info("=== Información del Sistema ===")
        logger.info(f"Memoria RAM: {info['memory_gb']:.1f} GB")
        logger.info(f"CPUs: {info['cpu_count']}")
        logger.info(f"GPU disponible: {info['gpu_available']}")
        if info['gpu_available']:
            logger.info(f"Número de GPUs: {info['gpu_count']}")
            logger.info(f"GPU actual: {torch.cuda.get_device_name()}")
        logger.info(f"Dispositivo seleccionado: {self.device}")
        logger.info("=" * 30)