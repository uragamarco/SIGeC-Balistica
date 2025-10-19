#!/usr/bin/env python3
"""
Sistema de Fallbacks
====================

Provee estrategias simples para manejar fallos de componentes y ejecutar
operaciones con valores de fallback o implementaciones alternativas.

Se integra de forma opcional con el registro central en `core.fallback_registry`.
"""

import logging
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Estrategias de fallback disponibles."""
    DISABLE = auto()
    MOCK = auto()
    REDUCE_FUNCTIONALITY = auto()


class FallbackSystem:
    """
    Orquestador de estrategias de fallback.

    - Permite registrar/consultar estrategias por componente.
    - Ejecuta operaciones aplicando fallback cuando ocurre una excepción.
    """

    def __init__(self):
        self._strategies = {}

    def register_strategy(self, component: str, strategy: FallbackStrategy) -> None:
        """Registra una estrategia de fallback para un componente."""
        self._strategies[component] = strategy

    def get_strategy(self, component: str) -> Optional[FallbackStrategy]:
        """Obtiene la estrategia registrada para un componente (si existe)."""
        return self._strategies.get(component)

    def execute_with_fallback(
        self,
        operation: Callable[..., Any],
        *args,
        fallback_value: Any = None,
        fallback_category: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Ejecuta `operation` y aplica fallback si ocurre una excepción.

        Args:
            operation: Función/llamable a ejecutar.
            *args, **kwargs: Argumentos para `operation`.
            fallback_value: Valor a devolver en caso de fallo (si se provee).
            fallback_category: Categoría de fallback del registro central (opcional).

        Returns:
            Resultado de `operation` o el fallback correspondiente.
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Operación fallida, aplicando fallback. {type(e).__name__}: {e}",
                exc_info=False,
            )

            # Si se especifica un valor de fallback, utilizarlo directamente
            if fallback_value is not None:
                return fallback_value

            # Si se especifica una categoría, intentar obtener un fallback del registro central
            if fallback_category:
                try:
                    from core.fallback_registry import get_fallback
                    fb = get_fallback(fallback_category)
                    if fb is not None:
                        return fb
                except Exception:
                    # No hacer hard-fail si el registro no está disponible
                    pass

            # Como último recurso, relanzar la excepción
            raise


__all__ = ["FallbackSystem", "FallbackStrategy"]