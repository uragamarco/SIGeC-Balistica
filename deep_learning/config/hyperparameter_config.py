"""
Configuración de hiperparámetros para SEACABAr Deep Learning

Este módulo define las clases de configuración para hiperparámetros,
optimización y schedulers de learning rate.

Author: SEACABAr Team
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import json
from pathlib import Path


@dataclass
class OptimizationConfig:
    """Configuración para optimizadores"""
    optimizer: str = "adam"  # adam, sgd, adamw, rmsprop
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # Para SGD
    betas: tuple = (0.9, 0.999)  # Para Adam/AdamW
    eps: float = 1e-8
    gradient_clip_norm: Optional[float] = None
    gradient_clip_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'betas': self.betas,
            'eps': self.eps,
            'gradient_clip_norm': self.gradient_clip_norm,
            'gradient_clip_value': self.gradient_clip_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Crea desde diccionario"""
        return cls(**data)


@dataclass
class SchedulerConfig:
    """Configuración para schedulers de learning rate"""
    scheduler: str = "cosine"  # cosine, step, exponential, plateau, none
    step_size: int = 30  # Para StepLR
    gamma: float = 0.1  # Factor de reducción
    T_max: int = 100  # Para CosineAnnealingLR
    eta_min: float = 1e-6  # LR mínimo para cosine
    patience: int = 10  # Para ReduceLROnPlateau
    factor: float = 0.5  # Factor de reducción para plateau
    warmup_epochs: int = 0  # Epochs de warmup
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'scheduler': self.scheduler,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'patience': self.patience,
            'factor': self.factor,
            'warmup_epochs': self.warmup_epochs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchedulerConfig':
        """Crea desde diccionario"""
        return cls(**data)


@dataclass
class HyperparameterConfig:
    """Configuración completa de hiperparámetros"""
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Hiperparámetros específicos del modelo
    dropout_rate: float = 0.2
    batch_norm: bool = True
    activation: str = "relu"  # relu, gelu, swish, leaky_relu
    
    # Regularización
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'optimization': self.optimization.to_dict(),
            'scheduler': self.scheduler.to_dict(),
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'activation': self.activation,
            'label_smoothing': self.label_smoothing,
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterConfig':
        """Crea desde diccionario"""
        optimization = OptimizationConfig.from_dict(data.get('optimization', {}))
        scheduler = SchedulerConfig.from_dict(data.get('scheduler', {}))
        
        return cls(
            optimization=optimization,
            scheduler=scheduler,
            dropout_rate=data.get('dropout_rate', 0.2),
            batch_norm=data.get('batch_norm', True),
            activation=data.get('activation', 'relu'),
            label_smoothing=data.get('label_smoothing', 0.0),
            mixup_alpha=data.get('mixup_alpha', 0.0),
            cutmix_alpha=data.get('cutmix_alpha', 0.0)
        )
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Guarda configuración en archivo JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'HyperparameterConfig':
        """Carga configuración desde archivo JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Configuraciones predefinidas
def create_cpu_optimized_config() -> HyperparameterConfig:
    """Configuración optimizada para CPU"""
    return HyperparameterConfig(
        optimization=OptimizationConfig(
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=1e-4
        ),
        scheduler=SchedulerConfig(
            scheduler="step",
            step_size=20,
            gamma=0.5
        ),
        dropout_rate=0.3,
        batch_norm=True,
        activation="relu"
    )


def create_gpu_optimized_config() -> HyperparameterConfig:
    """Configuración optimizada para GPU"""
    return HyperparameterConfig(
        optimization=OptimizationConfig(
            optimizer="adamw",
            learning_rate=0.003,
            weight_decay=1e-4,
            gradient_clip_norm=1.0
        ),
        scheduler=SchedulerConfig(
            scheduler="cosine",
            T_max=100,
            eta_min=1e-6,
            warmup_epochs=5
        ),
        dropout_rate=0.2,
        batch_norm=True,
        activation="gelu",
        label_smoothing=0.1,
        mixup_alpha=0.2
    )


def create_ballistic_classification_config() -> HyperparameterConfig:
    """Configuración para clasificación balística"""
    return HyperparameterConfig(
        optimization=OptimizationConfig(
            optimizer="adamw",
            learning_rate=0.001,
            weight_decay=1e-4,
            gradient_clip_norm=0.5
        ),
        scheduler=SchedulerConfig(
            scheduler="cosine",
            T_max=50,
            eta_min=1e-6,
            warmup_epochs=3
        ),
        dropout_rate=0.25,
        batch_norm=True,
        activation="relu",
        label_smoothing=0.05
    )


def create_ballistic_matching_config() -> HyperparameterConfig:
    """Configuración para matching balístico"""
    return HyperparameterConfig(
        optimization=OptimizationConfig(
            optimizer="adam",
            learning_rate=0.0005,
            weight_decay=1e-5,
            gradient_clip_norm=1.0
        ),
        scheduler=SchedulerConfig(
            scheduler="plateau",
            patience=8,
            factor=0.5,
            warmup_epochs=2
        ),
        dropout_rate=0.2,
        batch_norm=True,
        activation="relu"
    )


def quick_test():
    """Prueba rápida del módulo"""
    print("🧪 Probando configuración de hiperparámetros...")
    
    # Crear configuración
    config = create_ballistic_classification_config()
    print(f"✓ Configuración creada: {config.optimization.optimizer}")
    
    # Serialización
    data = config.to_dict()
    config2 = HyperparameterConfig.from_dict(data)
    print(f"✓ Serialización: {config2.optimization.learning_rate}")
    
    print("✅ Configuración de hiperparámetros funcionando")


if __name__ == "__main__":
    quick_test()