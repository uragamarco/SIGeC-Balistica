#!/usr/bin/env python3
"""
Sistema de Optimización de APIs para SIGeC-Balisticar.
Proporciona compresión automática, paginación inteligente, rate limiting y optimizaciones de rendimiento.
"""

import time
import gzip
import json
import asyncio
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
from functools import wraps
import aiohttp
import uvloop
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.compression import CompressionMiddleware
import orjson
import lz4.frame
import brotli

# Configurar logging
logger = logging.getLogger(__name__)

class CompressionType(Enum):
    """Tipos de compresión soportados."""
    NONE = "none"
    GZIP = "gzip"
    BROTLI = "br"
    LZ4 = "lz4"

class PaginationType(Enum):
    """Tipos de paginación."""
    OFFSET = "offset"
    CURSOR = "cursor"
    TOKEN = "token"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Estrategias de caché para APIs."""
    NO_CACHE = "no-cache"
    PRIVATE = "private"
    PUBLIC = "public"
    IMMUTABLE = "immutable"

@dataclass
class RateLimitConfig:
    """Configuración de rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    enable_sliding_window: bool = True

@dataclass
class CompressionConfig:
    """Configuración de compresión."""
    min_size_bytes: int = 1024
    compression_level: int = 6
    auto_detect: bool = True
    preferred_types: List[CompressionType] = field(default_factory=lambda: [
        CompressionType.BROTLI, CompressionType.GZIP, CompressionType.LZ4
    ])

@dataclass
class PaginationConfig:
    """Configuración de paginación."""
    default_page_size: int = 50
    max_page_size: int = 1000
    enable_cursor_pagination: bool = True
    enable_total_count: bool = True
    cursor_field: str = "id"

@dataclass
class APIMetrics:
    """Métricas de API."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    compression_ratio: float = 0.0
    cache_hit_rate: float = 0.0
    rate_limit_hits: int = 0
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

@dataclass
class RequestContext:
    """Contexto de solicitud."""
    request_id: str
    client_ip: str
    user_agent: str
    accept_encoding: List[str]
    start_time: float
    endpoint: str
    method: str
    size_bytes: int = 0
    compressed: bool = False
    cached: bool = False

class RateLimiter:
    """Sistema de rate limiting avanzado."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Verificar si la solicitud está permitida."""
        
        with self._lock:
            now = time.time()
            client_requests = self._requests[client_id]
            
            # Limpiar solicitudes antiguas
            self._cleanup_old_requests(client_requests, now)
            
            # Verificar límites
            minute_requests = sum(1 for t in client_requests if now - t <= 60)
            hour_requests = sum(1 for t in client_requests if now - t <= 3600)
            day_requests = sum(1 for t in client_requests if now - t <= 86400)
            
            # Verificar límites
            if minute_requests >= self.config.requests_per_minute:
                return False, {
                    'error': 'Rate limit exceeded',
                    'limit': 'per_minute',
                    'retry_after': 60 - (now - max(client_requests))
                }
            
            if hour_requests >= self.config.requests_per_hour:
                return False, {
                    'error': 'Rate limit exceeded',
                    'limit': 'per_hour',
                    'retry_after': 3600 - (now - min(client_requests))
                }
            
            if day_requests >= self.config.requests_per_day:
                return False, {
                    'error': 'Rate limit exceeded',
                    'limit': 'per_day',
                    'retry_after': 86400 - (now - min(client_requests))
                }
            
            # Registrar solicitud
            client_requests.append(now)
            
            return True, {
                'remaining_minute': self.config.requests_per_minute - minute_requests - 1,
                'remaining_hour': self.config.requests_per_hour - hour_requests - 1,
                'remaining_day': self.config.requests_per_day - day_requests - 1
            }
    
    def _cleanup_old_requests(self, requests: deque, now: float):
        """Limpiar solicitudes antiguas."""
        
        # Mantener solo las solicitudes del último día
        while requests and now - requests[0] > 86400:
            requests.popleft()

class ResponseCompressor:
    """Sistema de compresión de respuestas."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    def should_compress(self, content: bytes, accept_encoding: List[str]) -> bool:
        """Determinar si se debe comprimir la respuesta."""
        
        if len(content) < self.config.min_size_bytes:
            return False
        
        if not accept_encoding:
            return False
        
        # Verificar si el cliente soporta compresión
        supported_types = []
        for encoding in accept_encoding:
            if encoding in ['gzip', 'br', 'lz4']:
                supported_types.append(encoding)
        
        return len(supported_types) > 0
    
    def compress(self, content: bytes, accept_encoding: List[str]) -> Tuple[bytes, str]:
        """Comprimir contenido."""
        
        if not self.should_compress(content, accept_encoding):
            return content, 'identity'
        
        # Seleccionar mejor algoritmo
        compression_type = self._select_compression_type(accept_encoding)
        
        try:
            if compression_type == CompressionType.BROTLI and 'br' in accept_encoding:
                compressed = brotli.compress(content, quality=self.config.compression_level)
                return compressed, 'br'
            
            elif compression_type == CompressionType.GZIP and 'gzip' in accept_encoding:
                compressed = gzip.compress(content, compresslevel=self.config.compression_level)
                return compressed, 'gzip'
            
            elif compression_type == CompressionType.LZ4 and 'lz4' in accept_encoding:
                compressed = lz4.frame.compress(content)
                return compressed, 'lz4'
            
            else:
                return content, 'identity'
                
        except Exception as e:
            logger.warning(f"Error comprimiendo respuesta: {e}")
            return content, 'identity'
    
    def _select_compression_type(self, accept_encoding: List[str]) -> CompressionType:
        """Seleccionar mejor tipo de compresión."""
        
        for preferred in self.config.preferred_types:
            if preferred.value in accept_encoding or preferred.value == 'br' and 'br' in accept_encoding:
                return preferred
        
        return CompressionType.NONE

class SmartPaginator:
    """Sistema de paginación inteligente."""
    
    def __init__(self, config: PaginationConfig):
        self.config = config
    
    def paginate_data(self, data: List[Any], page: int = 1, page_size: Optional[int] = None,
                     cursor: Optional[str] = None, pagination_type: PaginationType = PaginationType.OFFSET) -> Dict[str, Any]:
        """Paginar datos."""
        
        page_size = min(page_size or self.config.default_page_size, self.config.max_page_size)
        
        if pagination_type == PaginationType.OFFSET:
            return self._offset_pagination(data, page, page_size)
        elif pagination_type == PaginationType.CURSOR:
            return self._cursor_pagination(data, cursor, page_size)
        elif pagination_type == PaginationType.HYBRID:
            return self._hybrid_pagination(data, page, page_size, cursor)
        else:
            return self._offset_pagination(data, page, page_size)
    
    def _offset_pagination(self, data: List[Any], page: int, page_size: int) -> Dict[str, Any]:
        """Paginación por offset."""
        
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_data = data[start_idx:end_idx]
        
        return {
            'data': paginated_data,
            'pagination': {
                'type': 'offset',
                'current_page': page,
                'page_size': page_size,
                'total_pages': total_pages,
                'total_items': total_items if self.config.enable_total_count else None,
                'has_next': page < total_pages,
                'has_previous': page > 1,
                'next_page': page + 1 if page < total_pages else None,
                'previous_page': page - 1 if page > 1 else None
            }
        }
    
    def _cursor_pagination(self, data: List[Any], cursor: Optional[str], page_size: int) -> Dict[str, Any]:
        """Paginación por cursor."""
        
        if not self.config.enable_cursor_pagination:
            return self._offset_pagination(data, 1, page_size)
        
        cursor_field = self.config.cursor_field
        start_idx = 0
        
        # Encontrar posición del cursor
        if cursor:
            try:
                cursor_value = json.loads(cursor)
                for i, item in enumerate(data):
                    if isinstance(item, dict) and item.get(cursor_field) == cursor_value:
                        start_idx = i + 1
                        break
            except Exception as e:
                logger.warning(f"Error procesando cursor: {e}")
        
        # Obtener datos paginados
        paginated_data = data[start_idx:start_idx + page_size]
        
        # Generar cursors
        next_cursor = None
        if len(paginated_data) == page_size and start_idx + page_size < len(data):
            last_item = paginated_data[-1]
            if isinstance(last_item, dict) and cursor_field in last_item:
                next_cursor = json.dumps(last_item[cursor_field])
        
        return {
            'data': paginated_data,
            'pagination': {
                'type': 'cursor',
                'page_size': page_size,
                'has_next': next_cursor is not None,
                'next_cursor': next_cursor,
                'cursor_field': cursor_field
            }
        }
    
    def _hybrid_pagination(self, data: List[Any], page: int, page_size: int, cursor: Optional[str]) -> Dict[str, Any]:
        """Paginación híbrida."""
        
        if cursor:
            return self._cursor_pagination(data, cursor, page_size)
        else:
            return self._offset_pagination(data, page, page_size)

class APIOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware de optimización de APIs."""
    
    def __init__(self, app, optimizer: 'APIOptimizer'):
        super().__init__(app)
        self.optimizer = optimizer
    
    async def dispatch(self, request: Request, call_next):
        """Procesar solicitud con optimizaciones."""
        
        start_time = time.time()
        
        # Crear contexto de solicitud
        context = RequestContext(
            request_id=self._generate_request_id(),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get('user-agent', ''),
            accept_encoding=self._parse_accept_encoding(request),
            start_time=start_time,
            endpoint=str(request.url.path),
            method=request.method
        )
        
        # Verificar rate limiting
        if not await self._check_rate_limit(context):
            return Response(
                content=orjson.dumps({'error': 'Rate limit exceeded'}),
                status_code=429,
                headers={'Content-Type': 'application/json'}
            )
        
        # Verificar caché
        cached_response = await self._check_cache(request, context)
        if cached_response:
            context.cached = True
            return cached_response
        
        # Procesar solicitud
        try:
            response = await call_next(request)
            
            # Optimizar respuesta
            optimized_response = await self._optimize_response(response, context)
            
            # Actualizar métricas
            await self._update_metrics(context, response.status_code, time.time() - start_time)
            
            return optimized_response
            
        except Exception as e:
            logger.error(f"Error procesando solicitud {context.request_id}: {e}")
            await self._update_metrics(context, 500, time.time() - start_time)
            raise
    
    def _generate_request_id(self) -> str:
        """Generar ID único de solicitud."""
        return hashlib.md5(f"{time.time()}{threading.current_thread().ident}".encode()).hexdigest()[:16]
    
    def _get_client_ip(self, request: Request) -> str:
        """Obtener IP del cliente."""
        forwarded = request.headers.get('x-forwarded-for')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'
    
    def _parse_accept_encoding(self, request: Request) -> List[str]:
        """Parsear Accept-Encoding header."""
        accept_encoding = request.headers.get('accept-encoding', '')
        return [enc.strip() for enc in accept_encoding.split(',') if enc.strip()]
    
    async def _check_rate_limit(self, context: RequestContext) -> bool:
        """Verificar rate limiting."""
        
        allowed, info = self.optimizer.rate_limiter.is_allowed(context.client_ip)
        
        if not allowed:
            self.optimizer.metrics.rate_limit_hits += 1
            logger.warning(f"Rate limit exceeded for {context.client_ip}: {info}")
        
        return allowed
    
    async def _check_cache(self, request: Request, context: RequestContext) -> Optional[Response]:
        """Verificar caché de respuesta."""
        
        # Implementación simplificada - en producción usar Redis o similar
        cache_key = f"{request.method}:{request.url.path}:{request.url.query}"
        
        # Por ahora, no implementamos caché de respuesta completo
        return None
    
    async def _optimize_response(self, response: Response, context: RequestContext) -> Response:
        """Optimizar respuesta."""
        
        # Leer contenido de respuesta
        if hasattr(response, 'body'):
            content = response.body
        else:
            # Para StreamingResponse y otros tipos
            return response
        
        if not content:
            return response
        
        # Comprimir si es necesario
        compressed_content, encoding = self.optimizer.compressor.compress(content, context.accept_encoding)
        
        if encoding != 'identity':
            context.compressed = True
            context.size_bytes = len(compressed_content)
            
            # Crear nueva respuesta comprimida
            headers = dict(response.headers)
            headers['content-encoding'] = encoding
            headers['content-length'] = str(len(compressed_content))
            
            return Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type
            )
        
        context.size_bytes = len(content)
        return response
    
    async def _update_metrics(self, context: RequestContext, status_code: int, response_time: float):
        """Actualizar métricas."""
        
        self.optimizer.metrics.total_requests += 1
        
        if 200 <= status_code < 400:
            self.optimizer.metrics.successful_requests += 1
        else:
            self.optimizer.metrics.failed_requests += 1
        
        # Actualizar tiempo promedio de respuesta
        current_avg = self.optimizer.metrics.avg_response_time_ms
        total_requests = self.optimizer.metrics.total_requests
        
        self.optimizer.metrics.avg_response_time_ms = (
            (current_avg * (total_requests - 1) + response_time * 1000) / total_requests
        )

class APIOptimizer:
    """Sistema principal de optimización de APIs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializar optimizador de APIs."""
        
        self.config = config or {}
        
        # Configuraciones
        self.rate_limit_config = RateLimitConfig(**self.config.get('rate_limiting', {}))
        self.compression_config = CompressionConfig(**self.config.get('compression', {}))
        self.pagination_config = PaginationConfig(**self.config.get('pagination', {}))
        
        # Componentes
        self.rate_limiter = RateLimiter(self.rate_limit_config)
        self.compressor = ResponseCompressor(self.compression_config)
        self.paginator = SmartPaginator(self.pagination_config)
        
        # Métricas
        self.metrics = APIMetrics()
        
        # Estado
        self._request_history: deque = deque(maxlen=10000)
        self._performance_samples: deque = deque(maxlen=1000)
        
        logger.info("Optimizador de APIs inicializado")
    
    def create_middleware(self) -> APIOptimizationMiddleware:
        """Crear middleware de optimización."""
        return lambda app: APIOptimizationMiddleware(app, self)
    
    def paginate(self, data: List[Any], **kwargs) -> Dict[str, Any]:
        """Paginar datos."""
        return self.paginator.paginate_data(data, **kwargs)
    
    def compress_response(self, content: bytes, accept_encoding: List[str]) -> Tuple[bytes, str]:
        """Comprimir respuesta."""
        return self.compressor.compress(content, accept_encoding)
    
    def check_rate_limit(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Verificar rate limit."""
        return self.rate_limiter.is_allowed(client_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'requests': {
                'total': self.metrics.total_requests,
                'successful': self.metrics.successful_requests,
                'failed': self.metrics.failed_requests,
                'success_rate': self.metrics.success_rate
            },
            'performance': {
                'avg_response_time_ms': self.metrics.avg_response_time_ms,
                'compression_ratio': self.metrics.compression_ratio,
                'cache_hit_rate': self.metrics.cache_hit_rate
            },
            'rate_limiting': {
                'hits': self.metrics.rate_limit_hits,
                'config': {
                    'requests_per_minute': self.rate_limit_config.requests_per_minute,
                    'requests_per_hour': self.rate_limit_config.requests_per_hour,
                    'requests_per_day': self.rate_limit_config.requests_per_day
                }
            },
            'compression': {
                'enabled': True,
                'min_size_bytes': self.compression_config.min_size_bytes,
                'preferred_types': [t.value for t in self.compression_config.preferred_types]
            },
            'pagination': {
                'default_page_size': self.pagination_config.default_page_size,
                'max_page_size': self.pagination_config.max_page_size,
                'cursor_enabled': self.pagination_config.enable_cursor_pagination
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Obtener reporte detallado de rendimiento."""
        
        if not self._performance_samples:
            return {'error': 'No hay datos de rendimiento disponibles'}
        
        response_times = list(self._performance_samples)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(response_times),
            'response_time_stats': {
                'min_ms': min(response_times),
                'max_ms': max(response_times),
                'avg_ms': sum(response_times) / len(response_times),
                'median_ms': sorted(response_times)[len(response_times) // 2],
                'p95_ms': sorted(response_times)[int(len(response_times) * 0.95)],
                'p99_ms': sorted(response_times)[int(len(response_times) * 0.99)]
            },
            'throughput': {
                'requests_per_second': len(response_times) / 60 if response_times else 0,  # Aproximado
                'total_requests': self.metrics.total_requests
            },
            'optimization_impact': {
                'compression_savings': self.metrics.compression_ratio,
                'cache_efficiency': self.metrics.cache_hit_rate,
                'rate_limit_effectiveness': self.metrics.rate_limit_hits / max(self.metrics.total_requests, 1)
            }
        }
    
    def optimize_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizar parámetros de consulta."""
        
        optimized = {}
        
        # Normalizar paginación
        if 'page' in params:
            optimized['page'] = max(1, int(params.get('page', 1)))
        
        if 'page_size' in params:
            optimized['page_size'] = min(
                max(1, int(params.get('page_size', self.pagination_config.default_page_size))),
                self.pagination_config.max_page_size
            )
        
        # Normalizar ordenamiento
        if 'sort' in params:
            sort_fields = params['sort'].split(',')
            valid_sorts = []
            for field in sort_fields:
                field = field.strip()
                if field.startswith('-'):
                    valid_sorts.append(f"-{field[1:]}")
                else:
                    valid_sorts.append(field)
            optimized['sort'] = ','.join(valid_sorts)
        
        # Normalizar filtros
        for key, value in params.items():
            if key not in ['page', 'page_size', 'sort'] and value is not None:
                optimized[key] = value
        
        return optimized
    
    def create_streaming_response(self, data_generator: AsyncGenerator, 
                                content_type: str = "application/json") -> StreamingResponse:
        """Crear respuesta de streaming optimizada."""
        
        async def generate_compressed_stream():
            """Generar stream comprimido."""
            
            buffer = []
            buffer_size = 0
            chunk_size = 8192  # 8KB chunks
            
            async for item in data_generator:
                # Serializar item
                if isinstance(item, (dict, list)):
                    serialized = orjson.dumps(item) + b'\n'
                else:
                    serialized = str(item).encode('utf-8') + b'\n'
                
                buffer.append(serialized)
                buffer_size += len(serialized)
                
                # Enviar chunk cuando alcance el tamaño objetivo
                if buffer_size >= chunk_size:
                    chunk = b''.join(buffer)
                    yield chunk
                    buffer = []
                    buffer_size = 0
            
            # Enviar último chunk
            if buffer:
                chunk = b''.join(buffer)
                yield chunk
        
        return StreamingResponse(
            generate_compressed_stream(),
            media_type=content_type,
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
    
    def add_performance_sample(self, response_time_ms: float):
        """Agregar muestra de rendimiento."""
        self._performance_samples.append(response_time_ms)
    
    def reset_metrics(self):
        """Resetear métricas."""
        self.metrics = APIMetrics()
        self._performance_samples.clear()
        logger.info("Métricas de API reseteadas")

# Decoradores de optimización

def optimize_endpoint(paginate: bool = True, compress: bool = True, 
                     cache_ttl: Optional[int] = None, rate_limit: bool = True):
    """Decorador para optimizar endpoints."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Ejecutar función
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Aplicar optimizaciones según configuración
                if paginate and isinstance(result, list):
                    # Aplicar paginación automática si el resultado es una lista
                    optimizer = get_api_optimizer()
                    result = optimizer.paginate(result)
                
                return result
                
            finally:
                # Registrar tiempo de respuesta
                response_time = (time.time() - start_time) * 1000
                optimizer = get_api_optimizer()
                optimizer.add_performance_sample(response_time)
        
        return wrapper
    return decorator

def require_rate_limit(requests_per_minute: int = 60):
    """Decorador para aplicar rate limiting específico."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            optimizer = get_api_optimizer()
            client_ip = request.client.host if request.client else 'unknown'
            
            allowed, info = optimizer.check_rate_limit(client_ip)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=info,
                    headers={'Retry-After': str(int(info.get('retry_after', 60)))}
                )
            
            return await func(request, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# Instancia global
_api_optimizer: Optional[APIOptimizer] = None

def get_api_optimizer() -> APIOptimizer:
    """Obtener instancia global del optimizador."""
    global _api_optimizer
    if _api_optimizer is None:
        _api_optimizer = APIOptimizer()
    return _api_optimizer

def initialize_api_optimizer(config: Dict[str, Any] = None) -> APIOptimizer:
    """Inicializar optimizador de APIs."""
    global _api_optimizer
    _api_optimizer = APIOptimizer(config)
    return _api_optimizer

# Utilidades de FastAPI

def create_optimized_app(config: Dict[str, Any] = None) -> FastAPI:
    """Crear aplicación FastAPI optimizada."""
    
    # Inicializar optimizador
    optimizer = initialize_api_optimizer(config)
    
    # Crear aplicación
    app = FastAPI(
        title="SIGeC-Balisticar API",
        description="API optimizada con compresión, paginación y rate limiting",
        version="1.0.0"
    )
    
    # Agregar middleware de optimización
    app.add_middleware(optimizer.create_middleware())
    
    # Agregar middleware de compresión nativo de FastAPI como respaldo
    app.add_middleware(CompressionMiddleware, minimum_size=1000)
    
    # Endpoints de métricas
    @app.get("/api/metrics")
    async def get_metrics():
        """Obtener métricas de API."""
        return optimizer.get_metrics()
    
    @app.get("/api/performance")
    async def get_performance():
        """Obtener reporte de rendimiento."""
        return optimizer.get_performance_report()
    
    @app.post("/api/metrics/reset")
    async def reset_metrics():
        """Resetear métricas."""
        optimizer.reset_metrics()
        return {"message": "Métricas reseteadas"}
    
    return app

if __name__ == "__main__":
    # Ejemplo de uso
    import uvicorn
    
    # Configuración de ejemplo
    config = {
        'rate_limiting': {
            'requests_per_minute': 100,
            'requests_per_hour': 2000,
            'requests_per_day': 20000
        },
        'compression': {
            'min_size_bytes': 512,
            'compression_level': 6,
            'preferred_types': ['brotli', 'gzip']
        },
        'pagination': {
            'default_page_size': 25,
            'max_page_size': 500,
            'enable_cursor_pagination': True
        }
    }
    
    # Crear aplicación optimizada
    app = create_optimized_app(config)
    
    # Endpoint de ejemplo
    @app.get("/api/data")
    @optimize_endpoint(paginate=True, compress=True)
    async def get_data(page: int = 1, page_size: int = 50):
        """Endpoint de ejemplo con optimizaciones."""
        
        # Simular datos
        data = [{"id": i, "name": f"Item {i}", "value": i * 10} for i in range(1, 1001)]
        
        return data
    
    # Endpoint con rate limiting específico
    @app.get("/api/heavy-operation")
    @require_rate_limit(requests_per_minute=10)
    async def heavy_operation():
        """Operación pesada con rate limiting estricto."""
        
        # Simular operación pesada
        await asyncio.sleep(1)
        
        return {"message": "Operación completada", "timestamp": datetime.now().isoformat()}
    
    # Ejecutar servidor
    print("Iniciando servidor API optimizado...")
    print("Métricas disponibles en: http://localhost:8000/api/metrics")
    print("Rendimiento disponible en: http://localhost:8000/api/performance")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="uvloop")