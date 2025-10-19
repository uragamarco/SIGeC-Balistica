#!/usr/bin/env python3
"""
Registro centralizado de fallbacks
==================================

Este módulo centraliza el acceso a las implementaciones de fallback y
utilidades relacionadas, exponiéndolas desde `core` para uso consistente
en toda la aplicación.

Mantiene compatibilidad reexportando clases y funciones desde
`utils.fallback_implementations`.
"""

from typing import Any, List

try:
    from utils.fallback_implementations import (
        DeepLearningFallback,
        WebServiceFallback,
        ImageProcessingFallback,
        DatabaseFallback,
        TorchFallback,
        TensorFlowFallback,
        FlaskFallback,
        RawPyFallback,
        CoreComponentsFallback,
        FallbackRegistry as _FallbackRegistry,
        get_fallback as _get_fallback,
        create_robust_import as _create_robust_import,
        fallback_registry as _fallback_registry,
    )
except Exception as e:  # pragma: no cover - solo en caso de entorno incompleto
    # Si por alguna razón las implementaciones no están disponibles,
    # definimos stubs mínimos para evitar fallos duros en importación.
    class _FallbackRegistry:  # type: ignore
        def __init__(self):
            self.fallbacks = {}
        def get_fallback(self, category: str) -> Any:
            return None
        def register_fallback(self, category: str, implementation: Any):
            self.fallbacks[category] = implementation
        def list_available_fallbacks(self) -> List[str]:
            return list(self.fallbacks.keys())
    def _get_fallback(category: str) -> Any:
        return None
    def _create_robust_import(package_name: str, fallback_category: str = None):
        def _import():
            raise ImportError(f"Paquete {package_name} no disponible")
        return _import
    _fallback_registry = _FallbackRegistry()


# Reexportar símbolos principales para uso desde core
FallbackRegistry = _FallbackRegistry
get_fallback = _get_fallback
create_robust_import = _create_robust_import
fallback_registry = _fallback_registry

__all__ = [
    # Clases de fallback
    "DeepLearningFallback",
    "WebServiceFallback",
    "ImageProcessingFallback",
    "DatabaseFallback",
    "TorchFallback",
    "TensorFlowFallback",
    "FlaskFallback",
    "RawPyFallback",
    "CoreComponentsFallback",
    # Registro y utilidades
    "FallbackRegistry",
    "get_fallback",
    "create_robust_import",
    "fallback_registry",
]