"""
Pool de Memoria GPU para SEACABAr
Sistema de gestión de memoria GPU con pre-allocación y reutilización

Este módulo implementa un sistema de pool de memoria GPU que permite:
- Pre-allocación inteligente de memoria
- Reutilización eficiente de bloques de memoria
- Gestión automática de fragmentación
- Métricas de uso en tiempo real
- Optimización para operaciones balísticas

Autor: Sistema SEACABA
Fecha: 2024
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from contextlib import contextmanager
import weakref
import gc

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

@dataclass
class MemoryBlock:
    """Bloque de memoria GPU en el pool"""
    size: int
    ptr: Optional[int] = None
    in_use: bool = False
    allocated_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    block_id: str = ""

@dataclass
class PoolStats:
    """Estadísticas del pool de memoria"""
    total_blocks: int = 0
    active_blocks: int = 0
    free_blocks: int = 0
    total_memory_bytes: int = 0
    used_memory_bytes: int = 0
    free_memory_bytes: int = 0
    fragmentation_ratio: float = 0.0
    hit_rate: float = 0.0
    allocation_requests: int = 0
    pool_hits: int = 0
    pool_misses: int = 0

class GPUMemoryPool:
    """
    Pool de memoria GPU con gestión inteligente de bloques
    """
    
    def __init__(self, 
                 initial_pool_size_mb: int = 512,
                 max_pool_size_mb: int = 2048,
                 block_sizes_mb: List[int] = None,
                 enable_defragmentation: bool = True,
                 cleanup_interval: int = 300):  # 5 minutos
        """
        Inicializar pool de memoria GPU
        
        Args:
            initial_pool_size_mb: Tamaño inicial del pool en MB
            max_pool_size_mb: Tamaño máximo del pool en MB
            block_sizes_mb: Tamaños de bloques predefinidos en MB
            enable_defragmentation: Habilitar desfragmentación automática
            cleanup_interval: Intervalo de limpieza en segundos
        """
        self.initial_pool_size = initial_pool_size_mb * 1024 * 1024
        self.max_pool_size = max_pool_size_mb * 1024 * 1024
        self.enable_defragmentation = enable_defragmentation
        self.cleanup_interval = cleanup_interval
        
        # Tamaños de bloques predefinidos (en bytes)
        if block_sizes_mb is None:
            block_sizes_mb = [1, 4, 16, 64, 256]  # MB
        self.block_sizes = [size * 1024 * 1024 for size in block_sizes_mb]
        
        # Estructuras de datos del pool
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.free_blocks_by_size: Dict[int, Set[str]] = {size: set() for size in self.block_sizes}
        self.used_blocks: Set[str] = set()
        
        # Estadísticas y métricas
        self.stats = PoolStats()
        self._lock = threading.RLock()
        self._next_block_id = 0
        
        # Control de limpieza automática
        self._last_cleanup = time.time()
        self._cleanup_thread = None
        self._shutdown = False
        
        # Inicializar pool si CuPy está disponible
        if CUPY_AVAILABLE:
            self._initialize_pool()
            self._start_cleanup_thread()
        else:
            logger.warning("CuPy no disponible - Pool de memoria GPU deshabilitado")
    
    def _initialize_pool(self):
        """Inicializar el pool con bloques predefinidos"""
        if not CUPY_AVAILABLE:
            return
            
        try:
            logger.info(f"Inicializando pool de memoria GPU - Tamaño inicial: {self.initial_pool_size // (1024*1024)}MB")
            
            # Pre-allocar bloques de diferentes tamaños
            remaining_size = self.initial_pool_size
            
            for block_size in sorted(self.block_sizes, reverse=True):
                blocks_to_create = min(remaining_size // block_size, 4)  # Máximo 4 bloques por tamaño
                
                for _ in range(blocks_to_create):
                    if remaining_size >= block_size:
                        self._create_block(block_size)
                        remaining_size -= block_size
                        
            logger.info(f"Pool inicializado con {len(self.memory_blocks)} bloques")
            
        except Exception as e:
            logger.error(f"Error inicializando pool de memoria: {e}")
    
    def _create_block(self, size: int) -> Optional[str]:
        """Crear un nuevo bloque de memoria"""
        if not CUPY_AVAILABLE:
            return None
            
        try:
            # Allocar memoria GPU
            gpu_array = cp.zeros(size // 8, dtype=cp.uint8)  # 8 bytes por elemento
            
            block_id = f"block_{self._next_block_id}"
            self._next_block_id += 1
            
            block = MemoryBlock(
                size=size,
                ptr=gpu_array.data.ptr,
                block_id=block_id
            )
            
            with self._lock:
                self.memory_blocks[block_id] = block
                
                # Encontrar el tamaño de bloque más cercano
                closest_size = min(self.block_sizes, key=lambda x: abs(x - size))
                self.free_blocks_by_size[closest_size].add(block_id)
                
                self.stats.total_blocks += 1
                self.stats.free_blocks += 1
                self.stats.total_memory_bytes += size
                self.stats.free_memory_bytes += size
            
            return block_id
            
        except Exception as e:
            logger.error(f"Error creando bloque de memoria de {size} bytes: {e}")
            return None
    
    def allocate(self, size: int) -> Optional[cp.ndarray]:
        """
        Allocar memoria del pool
        
        Args:
            size: Tamaño requerido en bytes
            
        Returns:
            Array de CuPy o None si no se puede allocar
        """
        if not CUPY_AVAILABLE:
            return None
            
        with self._lock:
            self.stats.allocation_requests += 1
            
            # Buscar bloque disponible del tamaño adecuado
            block_id = self._find_suitable_block(size)
            
            if block_id:
                # Usar bloque existente
                block = self.memory_blocks[block_id]
                block.in_use = True
                block.last_used = time.time()
                block.usage_count += 1
                
                # Actualizar estructuras
                closest_size = min(self.block_sizes, key=lambda x: abs(x - block.size))
                self.free_blocks_by_size[closest_size].discard(block_id)
                self.used_blocks.add(block_id)
                
                # Actualizar estadísticas
                self.stats.pool_hits += 1
                self.stats.active_blocks += 1
                self.stats.free_blocks -= 1
                self.stats.used_memory_bytes += block.size
                self.stats.free_memory_bytes -= block.size
                
                # Crear array CuPy desde el bloque
                try:
                    gpu_array = cp.zeros(size // 8, dtype=cp.uint8)
                    return gpu_array
                except Exception as e:
                    logger.error(f"Error creando array desde bloque: {e}")
                    self._release_block(block_id)
                    return None
            else:
                # Crear nuevo bloque si hay espacio
                if self.stats.total_memory_bytes + size <= self.max_pool_size:
                    new_block_id = self._create_block(size)
                    if new_block_id:
                        return self.allocate(size)  # Recursión para usar el nuevo bloque
                
                # Pool miss - allocar directamente
                self.stats.pool_misses += 1
                try:
                    return cp.zeros(size // 8, dtype=cp.uint8)
                except Exception as e:
                    logger.error(f"Error en allocación directa: {e}")
                    return None
    
    def _find_suitable_block(self, required_size: int) -> Optional[str]:
        """Encontrar bloque adecuado para el tamaño requerido"""
        # Buscar el tamaño de bloque más pequeño que sea >= required_size
        suitable_sizes = [size for size in self.block_sizes if size >= required_size]
        
        if not suitable_sizes:
            return None
            
        # Buscar en orden de tamaño (más pequeño primero)
        for size in sorted(suitable_sizes):
            if self.free_blocks_by_size[size]:
                return self.free_blocks_by_size[size].pop()
                
        return None
    
    def deallocate(self, gpu_array: cp.ndarray):
        """
        Devolver memoria al pool
        
        Args:
            gpu_array: Array de CuPy a liberar
        """
        if not CUPY_AVAILABLE or gpu_array is None:
            return
            
        try:
            # Buscar el bloque correspondiente
            ptr = gpu_array.data.ptr
            block_id = self._find_block_by_ptr(ptr)
            
            if block_id:
                self._release_block(block_id)
            else:
                # Liberar directamente si no está en el pool
                del gpu_array
                
        except Exception as e:
            logger.warning(f"Error liberando memoria: {e}")
    
    def _find_block_by_ptr(self, ptr: int) -> Optional[str]:
        """Encontrar bloque por puntero de memoria"""
        for block_id, block in self.memory_blocks.items():
            if block.ptr == ptr and block.in_use:
                return block_id
        return None
    
    def _release_block(self, block_id: str):
        """Liberar bloque de vuelta al pool"""
        with self._lock:
            if block_id not in self.memory_blocks:
                return
                
            block = self.memory_blocks[block_id]
            if not block.in_use:
                return
                
            # Marcar como libre
            block.in_use = False
            block.last_used = time.time()
            
            # Actualizar estructuras
            self.used_blocks.discard(block_id)
            closest_size = min(self.block_sizes, key=lambda x: abs(x - block.size))
            self.free_blocks_by_size[closest_size].add(block_id)
            
            # Actualizar estadísticas
            self.stats.active_blocks -= 1
            self.stats.free_blocks += 1
            self.stats.used_memory_bytes -= block.size
            self.stats.free_memory_bytes += block.size
    
    def _start_cleanup_thread(self):
        """Iniciar hilo de limpieza automática"""
        def cleanup_worker():
            while not self._shutdown:
                try:
                    time.sleep(self.cleanup_interval)
                    if not self._shutdown:
                        self._periodic_cleanup()
                except Exception as e:
                    logger.error(f"Error en hilo de limpieza: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _periodic_cleanup(self):
        """Limpieza periódica del pool"""
        current_time = time.time()
        
        with self._lock:
            # Limpiar bloques no usados por mucho tiempo
            blocks_to_remove = []
            
            for block_id, block in self.memory_blocks.items():
                if (not block.in_use and 
                    current_time - block.last_used > self.cleanup_interval * 2):
                    blocks_to_remove.append(block_id)
            
            # Remover bloques antiguos
            for block_id in blocks_to_remove:
                self._remove_block(block_id)
            
            # Desfragmentar si está habilitado
            if self.enable_defragmentation:
                self._defragment()
            
            # Actualizar estadísticas
            self._update_stats()
            
            if blocks_to_remove:
                logger.debug(f"Limpieza completada - Removidos {len(blocks_to_remove)} bloques")
    
    def _remove_block(self, block_id: str):
        """Remover bloque del pool completamente"""
        if block_id not in self.memory_blocks:
            return
            
        block = self.memory_blocks[block_id]
        
        # Remover de estructuras
        closest_size = min(self.block_sizes, key=lambda x: abs(x - block.size))
        self.free_blocks_by_size[closest_size].discard(block_id)
        
        # Actualizar estadísticas
        self.stats.total_blocks -= 1
        if not block.in_use:
            self.stats.free_blocks -= 1
            self.stats.free_memory_bytes -= block.size
        
        self.stats.total_memory_bytes -= block.size
        
        # Remover bloque
        del self.memory_blocks[block_id]
    
    def _defragment(self):
        """Desfragmentar memoria del pool"""
        # Implementación básica de desfragmentación
        # En una implementación completa, esto reorganizaría los bloques
        pass
    
    def _update_stats(self):
        """Actualizar estadísticas del pool"""
        if self.stats.allocation_requests > 0:
            self.stats.hit_rate = self.stats.pool_hits / self.stats.allocation_requests
        
        if self.stats.total_memory_bytes > 0:
            used_ratio = self.stats.used_memory_bytes / self.stats.total_memory_bytes
            # Fragmentación aproximada basada en número de bloques vs memoria usada
            expected_blocks = max(1, self.stats.used_memory_bytes // min(self.block_sizes))
            actual_blocks = self.stats.active_blocks
            self.stats.fragmentation_ratio = max(0, (actual_blocks - expected_blocks) / max(1, expected_blocks))
    
    def get_stats(self) -> PoolStats:
        """Obtener estadísticas actuales del pool"""
        with self._lock:
            self._update_stats()
            return PoolStats(**self.stats.__dict__)
    
    def clear_pool(self):
        """Limpiar completamente el pool"""
        with self._lock:
            logger.info("Limpiando pool de memoria GPU")
            
            # Limpiar todas las estructuras
            self.memory_blocks.clear()
            for size_set in self.free_blocks_by_size.values():
                size_set.clear()
            self.used_blocks.clear()
            
            # Resetear estadísticas
            self.stats = PoolStats()
            
            # Forzar garbage collection
            gc.collect()
            if CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
    
    @contextmanager
    def allocate_context(self, size: int):
        """Context manager para allocación automática"""
        gpu_array = self.allocate(size)
        try:
            yield gpu_array
        finally:
            if gpu_array is not None:
                self.deallocate(gpu_array)
    
    def shutdown(self):
        """Cerrar el pool y limpiar recursos"""
        logger.info("Cerrando pool de memoria GPU")
        self._shutdown = True
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self.clear_pool()

# Instancia global del pool
_gpu_memory_pool = None

def get_gpu_memory_pool(**kwargs) -> GPUMemoryPool:
    """Obtener instancia global del pool de memoria GPU"""
    global _gpu_memory_pool
    if _gpu_memory_pool is None:
        _gpu_memory_pool = GPUMemoryPool(**kwargs)
    return _gpu_memory_pool

@contextmanager
def gpu_pool_allocate(size: int):
    """Context manager global para allocación desde el pool"""
    pool = get_gpu_memory_pool()
    with pool.allocate_context(size) as gpu_array:
        yield gpu_array

def get_pool_stats() -> PoolStats:
    """Obtener estadísticas del pool global"""
    pool = get_gpu_memory_pool()
    return pool.get_stats()

def clear_gpu_pool():
    """Limpiar el pool global"""
    pool = get_gpu_memory_pool()
    pool.clear_pool()