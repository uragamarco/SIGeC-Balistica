"""
Configuraciones de entrenamiento optimizadas para el dataset NIST FADB
Incluye configuraciones para diferentes escenarios y tamaños de dataset
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from .experiment_config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig

def create_nist_fadb_config(
    dataset_size: str = "full",
    model_complexity: str = "medium",
    training_mode: str = "classification"
) -> ExperimentConfig:
    """
    Crear configuración optimizada para el dataset NIST FADB
    
    Args:
        dataset_size: "small" (<1000 imgs), "medium" (1000-5000), "full" (>5000)
        model_complexity: "light", "medium", "heavy"
        training_mode: "classification", "matching", "both"
    """
    
    # Configuraciones base según el tamaño del dataset
    size_configs = {
        "small": {
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.001,
            "backbone_feature_dim": 256,
            "feature_dim": 128
        },
        "medium": {
            "batch_size": 32,
            "epochs": 75,
            "learning_rate": 0.0005,
            "backbone_feature_dim": 512,
            "feature_dim": 256
        },
        "full": {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.0001,
            "backbone_feature_dim": 1024,
            "feature_dim": 512
        }
    }
    
    # Configuraciones según complejidad del modelo
    complexity_configs = {
        "light": {
            "backbone_type": "SimpleCNN",
            "use_attention": False,
            "dropout_rate": 0.3
        },
        "medium": {
            "backbone_type": "BallisticCNN",
            "use_attention": True,
            "dropout_rate": 0.4
        },
        "heavy": {
            "backbone_type": "ResNetBackbone",
            "use_attention": True,
            "dropout_rate": 0.5
        }
    }
    
    base_config = size_configs[dataset_size]
    model_config = complexity_configs[model_complexity]
    
    # Configuración del modelo
    model_cfg = ModelConfig(
        model_type="BallisticCNN",
        architecture="custom",  # Usar arquitectura custom que está soportada
        num_classes=13,  # Basado en el análisis del dataset NIST FADB
        feature_dim=base_config["feature_dim"],
        use_attention=model_config["use_attention"],
        dropout_rate=model_config["dropout_rate"]
    )
    
    # Configuración de datos
    data_cfg = DataConfig(
        data_root="uploads/Muestras NIST FADB",
        target_size=(224, 224),
        batch_size=base_config["batch_size"],
        num_workers=4,
        pin_memory=True,
        use_augmentation=True,
        rotation_range=15.0,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        noise_std=0.01,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        balance_classes=True
    )
    
    # Configuración de entrenamiento
    training_cfg = TrainingConfig(
        epochs=base_config["epochs"],
        learning_rate=base_config["learning_rate"],
        weight_decay=1e-4,
        optimizer="adam",
        use_scheduler=True,
        scheduler_type="plateau",
        patience=5,
        use_early_stopping=True,
        early_stopping_patience=10,
        save_checkpoints=True,
        save_best_only=True,
        validate_frequency=1
    )
    
    # Configuración de evaluación
    eval_cfg = EvaluationConfig(
        primary_metric="accuracy",
        compute_confusion_matrix=True,
        compute_roc_curves=True,
        per_class_metrics=True,
        use_cross_validation=True,  # Habilitado con menos folds
        cv_folds=3,
        save_plots=True,
        generate_html_report=True,
        include_sample_predictions=True
    )
    
    return ExperimentConfig(
        model=model_cfg,
        data=data_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
        experiment_name=f"nist_fadb_{dataset_size}_{model_complexity}_{training_mode}",
        description=f"Configuración optimizada para NIST FADB - {dataset_size} dataset, {model_complexity} model complexity"
    )

def create_quick_test_config() -> ExperimentConfig:
    """Configuración para pruebas rápidas con subset pequeño del dataset"""
    return create_nist_fadb_config(
        dataset_size="small",
        model_complexity="light",
        training_mode="classification"
    )

def create_production_config() -> ExperimentConfig:
    """Configuración para entrenamiento de producción con dataset completo"""
    return create_nist_fadb_config(
        dataset_size="full",
        model_complexity="medium",
        training_mode="classification"
    )

def create_research_config() -> ExperimentConfig:
    """Configuración para investigación con modelo complejo"""
    return create_nist_fadb_config(
        dataset_size="full",
        model_complexity="heavy",
        training_mode="both"
    )

def get_recommended_config(num_images: int, available_memory_gb: float) -> ExperimentConfig:
    """
    Obtener configuración recomendada basada en el tamaño del dataset y memoria disponible
    
    Args:
        num_images: Número total de imágenes en el dataset
        available_memory_gb: Memoria RAM disponible en GB
    """
    
    # Determinar tamaño del dataset
    if num_images < 1000:
        dataset_size = "small"
    elif num_images < 5000:
        dataset_size = "medium"
    else:
        dataset_size = "full"
    
    # Determinar complejidad del modelo basada en memoria disponible
    if available_memory_gb < 4:
        model_complexity = "light"
    elif available_memory_gb < 8:
        model_complexity = "medium"
    else:
        model_complexity = "heavy"
    
    config = create_nist_fadb_config(dataset_size, model_complexity, "classification")
    
    # Ajustar batch size según memoria disponible
    if available_memory_gb < 4:
        config.data.batch_size = min(config.data.batch_size, 16)
    elif available_memory_gb < 8:
        config.data.batch_size = min(config.data.batch_size, 32)
    
    return config

# Configuraciones predefinidas para casos específicos
NIST_FADB_CONFIGS = {
    "quick_test": create_quick_test_config(),
    "production": create_production_config(),
    "research": create_research_config()
}

def list_available_configs() -> Dict[str, str]:
    """Listar configuraciones disponibles con sus descripciones"""
    return {
        "quick_test": "Configuración rápida para pruebas con modelo ligero",
        "production": "Configuración optimizada para producción",
        "research": "Configuración completa para investigación",
        "custom": "Configuración personalizada basada en parámetros específicos"
    }

def validate_config_for_dataset(config: ExperimentConfig, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validar y ajustar configuración según las estadísticas del dataset
    
    Args:
        config: Configuración a validar
        dataset_stats: Estadísticas del dataset (de NISTFADBLoader.get_dataset_statistics())
    
    Returns:
        Dict con recomendaciones y ajustes
    """
    recommendations = {
        "warnings": [],
        "suggestions": [],
        "adjustments": {}
    }
    
    total_images = dataset_stats.get("total_images", 0)
    num_studies = dataset_stats.get("total_studies", 0)  # Usar total_studies en lugar de num_studies
    
    # Validar número de clases
    if config.model.num_classes != num_studies:
        recommendations["adjustments"]["num_classes"] = num_studies
        recommendations["warnings"].append(
            f"Número de clases en config ({config.model.num_classes}) "
            f"no coincide con estudios en dataset ({num_studies})"
        )
    
    # Validar tamaño de batch vs dataset
    images_per_study = dataset_stats.get("distribution_by_study", {})
    if images_per_study:
        min_samples_per_class = min(images_per_study.values())
        if config.data.batch_size > min_samples_per_class:
            recommended_batch = max(8, min_samples_per_class // 2)
            recommendations["adjustments"]["batch_size"] = recommended_batch
            recommendations["warnings"].append(
                f"Batch size ({config.data.batch_size}) muy grande para la clase más pequeña "
                f"({min_samples_per_class} muestras). Recomendado: {recommended_batch}"
            )
    else:
        # Si no hay datos, usar configuración conservadora
        if config.data.batch_size > 16:
            recommendations["adjustments"]["batch_size"] = 16
            recommendations["warnings"].append(
                "No hay datos disponibles. Usando batch size conservador de 16."
            )
    
    # Sugerir épocas según tamaño del dataset
    if total_images < 500 and config.training.epochs < 100:
        recommendations["suggestions"].append(
            "Dataset pequeño: considerar aumentar épocas para mejor convergencia"
        )
    elif total_images > 5000 and config.training.epochs > 50:
        recommendations["suggestions"].append(
            "Dataset grande: considerar reducir épocas para evitar sobreentrenamiento"
        )
    
    return recommendations