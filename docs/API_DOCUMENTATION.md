---
title: Documentación de API
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: desarrolladores
toc: true
tags:
  - api
  - matching
  - pipeline
  - configuración
  - base_de_datos
---

# SIGeC-Balisticar - Documentación de API

## Overview

SIGeC-Balisticar is a comprehensive ballistic analysis system that provides advanced image processing, feature extraction, and comparison capabilities for forensic applications. This document provides detailed API documentation for all public interfaces and main modules.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Image Processing](#image-processing)
3. [Matching and Analysis](#matching-and-analysis)
4. [Database Management](#database-management)
5. [NIST Standards](#nist-standards)
6. [Configuration](#configuration)
7. [GUI Components](#gui-components)
8. [Interfaces](#interfaces)
9. [Utilities](#utilities)
10. [Examples](#examples)
11. [REST API Endpoints](#rest-api-endpoints)
12. [Error Handling](#error-handling)
13. [Configuration Examples](#configuration-examples)

## Core Modules

### ScientificPipeline

The main processing pipeline that orchestrates the entire ballistic analysis workflow.

```python
from core.unified_pipeline import ScientificPipeline, PipelineConfiguration

# Initialize pipeline
config = PipelineConfiguration(level="forensic")
pipeline = ScientificPipeline(config)

# Process image comparison
result = pipeline.process_comparison("image1.jpg", "image2.jpg")

# Interface methods (IPipelineProcessor)
success = pipeline.initialize(config_dict)
result = pipeline.process_images("path1", "path2")
capabilities = pipeline.get_capabilities()
pipeline.cleanup()
```

**Key Methods:**
- `process_comparison(image1, image2) -> PipelineResult`: Main processing method
- `initialize(config) -> bool`: Initialize with configuration
- `process_images(path1, path2) -> ProcessingResult`: Interface-compliant processing
- `get_capabilities() -> Dict[str, bool]`: Get system capabilities
- `cleanup()`: Clean up resources

### Performance Monitor

System performance monitoring and optimization.

```python
from core.performance_monitor import monitor_performance, OperationType

@monitor_performance(OperationType.IMAGE_PROCESSING)
def process_image(image):
    # Your processing code here
    pass
```

### Error Handler

Centralized error handling and recovery system.

```python
from core.error_handler import ErrorHandler, ErrorSeverity

handler = ErrorHandler()
handler.handle_error(exception, ErrorSeverity.HIGH, context="image_processing")
```

---

## Image Processing

### UnifiedPreprocessor

Advanced image preprocessing with NIST compliance.

```python
from image_processing.unified_preprocessor import UnifiedPreprocessor, PreprocessingConfig

config = PreprocessingConfig(level="forensic")
preprocessor = UnifiedPreprocessor(config)

# Process image
processed_image = preprocessor.process_image(image)
quality_report = preprocessor.get_quality_report()
```

### UnifiedROIDetector

Region of Interest detection using advanced algorithms.

```python
from image_processing.unified_roi_detector import UnifiedROIDetector, ROIDetectionConfig

config = ROIDetectionConfig(level="advanced")
detector = UnifiedROIDetector(config)

# Detect ROI
roi_regions = detector.detect_roi(image)
visualization = detector.visualize_roi(image, roi_regions)
```

### FeatureExtractor

Extract ballistic features from images.

```python
from image_processing.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Extract features
features = extractor.extract_features(image)
keypoints = extractor.extract_keypoints(image)
descriptors = extractor.extract_descriptors(image, keypoints)
```

---

## Matching and Analysis

### UnifiedMatcher

Advanced feature matching with multiple algorithms.

```python
from matching.unified_matcher import UnifiedMatcher, MatchingConfig, AlgorithmType

config = MatchingConfig(algorithm=AlgorithmType.SIFT)
matcher = UnifiedMatcher(config)

# Interface methods (IFeatureMatcher)
features1 = matcher.extract_features(image1)
features2 = matcher.extract_features(image2)
matches = matcher.match_features(features1, features2)
similarity = matcher.calculate_similarity(matches)
info = matcher.get_algorithm_info()
```

### CMC Algorithm

Consecutive Matching Characteristics analysis.

```python
from matching.cmc_algorithm import CMCAlgorithm, CMCParameters

params = CMCParameters(min_cmc_count=6, confidence_threshold=0.8)
cmc = CMCAlgorithm(params)

# Perform CMC analysis
result = cmc.analyze_matches(matches, image1, image2)
cmc_count = result.cmc_count
confidence = result.confidence
```

---

## Database Management

### UnifiedDatabase

Comprehensive database management with vector search capabilities.

```python
from database.unified_database import UnifiedDatabase

# Interface methods (IDatabaseManager)
db = UnifiedDatabase()
success = db.connect(connection_params)
db.store_analysis(analysis_id, data)
results = db.retrieve_analysis(analysis_id)
similar_cases = db.search_similar_cases(query_data)
stats = db.get_statistics()
db.backup_database(backup_path)
db.close_connection()

# Additional methods
db.add_case(case_data)
db.add_image(image_data)
cases = db.get_cases()
images = db.get_images_by_case(case_id)
```

### VectorDatabase

High-performance vector similarity search.

```python
from database.vector_db import VectorDatabase

vector_db = VectorDatabase()
vector_db.add_vector(vector_id, feature_vector, metadata)
similar_vectors = vector_db.search_similar(query_vector, top_k=10)
```

---

## NIST Standards

### Quality Metrics

NIST-compliant image quality assessment.

```python
from nist_standards.quality_metrics import NISTQualityMetrics

metrics = NISTQualityMetrics()
quality_report = metrics.assess_image_quality(image)

# Quality metrics
snr = quality_report.snr_value
contrast = quality_report.contrast_value
sharpness = quality_report.sharpness_value
overall_quality = quality_report.overall_quality
```

### AFTE Conclusions

Automated AFTE conclusion generation.

```python
from nist_standards.afte_conclusions import AFTEConclusionEngine, AFTEConclusion

engine = AFTEConclusionEngine()
conclusion = engine.determine_conclusion(similarity_score, confidence, cmc_count)

# Possible conclusions
if conclusion == AFTEConclusion.IDENTIFICATION:
    print("Positive identification")
elif conclusion == AFTEConclusion.INCONCLUSIVE:
    print("Inconclusive result")
```

---

## Configuration

### Unified Configuration

Centralized configuration management.

```python
from config.unified_config import UnifiedConfig

config = UnifiedConfig()
config.load_config("config.yaml")

# Access configuration values
processing_level = config.get("processing.level", "standard")
gpu_enabled = config.get("hardware.gpu_enabled", False)

# Update configuration
config.set("processing.algorithm", "sift")
config.save_config("updated_config.yaml")
```

---

## GUI Components

### Main Window

Primary GUI application window.

```python
from gui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
```

### Backend Integration

GUI-backend communication layer.

```python
from gui.backend_integration import BackendIntegration

backend = BackendIntegration()

# System status
status = backend.get_system_status()
formats = backend.get_supported_formats()
algorithms = backend.get_processing_algorithms()

# Analysis operations
result = backend.analyze_image(image_path, config)
comparison = backend.compare_images(image1_path, image2_path, config)
search_results = backend.search_database(query_image, filters)
```

---

## Interfaces

### IPipelineProcessor

Interface for pipeline processors.

```python
from interfaces.pipeline_interfaces import IPipelineProcessor, ProcessingResult

class CustomProcessor(IPipelineProcessor):
    def initialize(self, config: Dict[str, Any]) -> bool:
        # Implementation
        pass
    
    def process_images(self, image1_path: str, image2_path: str) -> ProcessingResult:
        # Implementation
        pass
    
    def get_capabilities(self) -> Dict[str, bool]:
        # Implementation
        pass
    
    def cleanup(self):
        # Implementation
        pass
```

### IFeatureMatcher

Interface for feature matching algorithms.

```python
from interfaces.matcher_interfaces import IFeatureMatcher

class CustomMatcher(IFeatureMatcher):
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        # Implementation
        pass
    
    def match_features(self, features1: Dict, features2: Dict) -> List[Dict]:
        # Implementation
        pass
    
    def calculate_similarity(self, matches: List[Dict]) -> float:
        # Implementation
        pass
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        # Implementation
        pass
```

### IDatabaseManager

Interface for database management.

```python
from interfaces.database_interfaces import IDatabaseManager

class CustomDatabase(IDatabaseManager):
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        # Implementation
        pass
    
    def store_analysis(self, analysis_id: str, data: Dict[str, Any]) -> bool:
        # Implementation
        pass
    
    # ... other methods
```

---

## Utilities

### Logger

Centralized logging system.

```python
from utils.logger import get_logger, LoggerMixin

# Get logger instance
logger = get_logger("my_module")
logger.info("Processing started")
logger.error("Error occurred", exc_info=True)

# Use as mixin
class MyClass(LoggerMixin):
    def process(self):
        self.logger.info("Processing in MyClass")
```

### Dependency Manager

Manage system dependencies and fallbacks.

```python
from utils.dependency_manager import DependencyManager

manager = DependencyManager()
available_deps = manager.check_dependencies()
missing_deps = manager.get_missing_dependencies()
```

---

## Examples

### Basic Image Analysis

```python
from core.unified_pipeline import ScientificPipeline, create_pipeline_config

# Create pipeline with forensic-level configuration
config = create_pipeline_config("forensic")
pipeline = ScientificPipeline(config)

# Analyze two ballistic images
result = pipeline.process_comparison("bullet1.jpg", "bullet2.jpg")

# Check results
print(f"AFTE Conclusion: {result.afte_conclusion.value}")
print(f"Similarity Score: {result.similarity_score:.3f}")
print(f"CMC Count: {result.cmc_count}")
print(f"Confidence: {result.confidence:.3f}")

# Export detailed report
pipeline.export_report(result, "analysis_report.json")
```

### Custom Processing Pipeline

```python
from interfaces.pipeline_interfaces import IPipelineProcessor, ProcessingResult
from core.unified_pipeline import ScientificPipeline

class CustomBalisticProcessor(IPipelineProcessor):
    def __init__(self):
        self.pipeline = ScientificPipeline()
    
    def initialize(self, config):
        return self.pipeline.initialize(config)
    
    def process_images(self, image1_path, image2_path):
        # Custom preprocessing
        result = self.pipeline.process_comparison(image1_path, image2_path)
        
        # Convert to interface format
        return ProcessingResult(
            success=len(result.error_messages) == 0,
            similarity_score=result.similarity_score,
            quality_score=result.quality_weighted_score,
            processing_time=result.processing_time,
            metadata={'afte_conclusion': result.afte_conclusion.value}
        )
    
    def get_capabilities(self):
        return self.pipeline.get_capabilities()
    
    def cleanup(self):
        self.pipeline.cleanup()

# Usage
processor = CustomBalisticProcessor()
processor.initialize({'level': 'forensic'})
result = processor.process_images('image1.jpg', 'image2.jpg')
```

### Database Integration

```python
from database.unified_database import UnifiedDatabase

# Initialize database
db = UnifiedDatabase()
db.connect({'database_path': 'ballistic_cases.db'})

# Add a new case
case_data = {
    'case_id': 'CASE_2024_001',
    'description': 'Ballistic analysis case',
    'date_created': '2024-01-15',
    'investigator': 'John Doe'
}
db.add_case(case_data)

# Add images to case
image_data = {
    'image_id': 'IMG_001',
    'case_id': 'CASE_2024_001',
    'image_path': 'evidence/bullet1.jpg',
    'metadata': {'caliber': '9mm', 'weapon_type': 'pistol'}
}
db.add_image(image_data)

# Search for similar cases
query_data = {'feature_vector': feature_vector, 'threshold': 0.8}
similar_cases = db.search_similar_cases(query_data)

# Get statistics
stats = db.get_statistics()
print(f"Total cases: {stats['total_cases']}")
print(f"Total images: {stats['total_images']}")

db.close_connection()
```

---

## REST API Endpoints

### Authentication

#### 1. API Key

```http
GET /api/v1/samples
Authorization: Bearer YOUR_API_KEY
```

#### 2. JWT Token

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "usuario",
  "password": "contraseña"
}
```

**Respuesta**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 3. Renovar Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json
Authorization: Bearer REFRESH_TOKEN

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Obtener API Key

```http
POST /api/v1/auth/api-keys
Authorization: Bearer JWT_TOKEN
Content-Type: application/json

{
  "name": "Mi Aplicación",
  "permissions": ["read", "write"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

---

## Endpoints de Análisis

### Analizar Imagen

Procesa una imagen y extrae características balísticas.

```http
POST /api/v1/analysis/process
Authorization: Bearer YOUR_TOKEN
Content-Type: multipart/form-data

{
  "image": [archivo_imagen],
  "algorithm": "lbp",
  "parameters": {
    "radius": 2,
    "points": 16
  },
  "metadata": {
    "case_id": "CASO_2024_001",
    "investigator": "Juan Pérez",
    "sample_type": "casquillo"
  }
}
```

**Respuesta**:
```json
{
  "id": "analysis_12345",
  "status": "completed",
  "sample_id": "BAL_2024_001",
  "algorithm": "lbp",
  "features": {
    "dimensions": 256,
    "histogram": [0.1, 0.2, 0.15, ...],
    "statistics": {
      "mean": 0.125,
      "std": 0.045,
      "entropy": 4.23
    }
  },
  "processing_time": 2.3,
  "created_at": "2024-01-15T14:30:25Z"
}
```

### Obtener Estado de Análisis

```http
GET /api/v1/analysis/{analysis_id}
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "id": "analysis_12345",
  "status": "processing",
  "progress": 75,
  "estimated_completion": "2024-01-15T14:32:00Z",
  "current_step": "feature_extraction"
}
```

### Listar Análisis

```http
GET /api/v1/analysis?page=1&limit=20&status=completed
Authorization: Bearer YOUR_TOKEN
```

**Parámetros de Consulta**:
- `page`: Número de página (default: 1)
- `limit`: Elementos por página (default: 20, max: 100)
- `status`: Filtrar por estado (`pending`, `processing`, `completed`, `failed`)
- `algorithm`: Filtrar por algoritmo (`lbp`, `sift`, `orb`)
- `date_from`: Fecha desde (ISO 8601)
- `date_to`: Fecha hasta (ISO 8601)

**Respuesta**:
```json
{
  "data": [
    {
      "id": "analysis_12345",
      "sample_id": "BAL_2024_001",
      "status": "completed",
      "algorithm": "lbp",
      "created_at": "2024-01-15T14:30:25Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

### Configurar Algoritmos

```http
PUT /api/v1/analysis/algorithms/{algorithm}
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "parameters": {
    "radius": 3,
    "points": 24,
    "uniform": true
  },
  "preprocessing": {
    "normalize_illumination": true,
    "reduce_noise": true,
    "enhance_contrast": false
  }
}
```

---

## Endpoints de Base de Datos

### Crear Muestra

```http
POST /api/v1/samples
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "name": "Muestra_001",
  "type": "casquillo",
  "description": "Casquillo encontrado en escena del crimen",
  "metadata": {
    "case_id": "CASO_2024_001",
    "investigator": "Juan Pérez",
    "collection_date": "2024-01-15",
    "location": "Escena A"
  }
}
```

**Respuesta**:
```json
{
  "id": "BAL_2024_001",
  "name": "Muestra_001",
  "type": "casquillo",
  "status": "pending",
  "created_at": "2024-01-15T14:30:25Z",
  "updated_at": "2024-01-15T14:30:25Z"
}
```

### Obtener Muestra

```http
GET /api/v1/samples/{sample_id}
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "id": "BAL_2024_001",
  "name": "Muestra_001",
  "type": "casquillo",
  "status": "processed",
  "description": "Casquillo encontrado en escena del crimen",
  "image_url": "https://storage.sigec.com/images/BAL_2024_001.jpg",
  "features": {
    "algorithm": "lbp",
    "dimensions": 256,
    "extracted_at": "2024-01-15T14:32:15Z"
  },
  "metadata": {
    "case_id": "CASO_2024_001",
    "investigator": "Juan Pérez",
    "collection_date": "2024-01-15",
    "location": "Escena A"
  },
  "created_at": "2024-01-15T14:30:25Z",
  "updated_at": "2024-01-15T14:32:15Z"
}
```

### Listar Muestras

```http
GET /api/v1/samples?page=1&limit=20&type=casquillo&status=processed
Authorization: Bearer YOUR_TOKEN
```

**Parámetros de Consulta**:
- `page`: Número de página
- `limit`: Elementos por página
- `type`: Tipo de muestra (`casquillo`, `proyectil`, `arma`)
- `status`: Estado (`pending`, `processing`, `processed`, `failed`)
- `case_id`: ID del caso
- `investigator`: Nombre del investigador
- `search`: Búsqueda de texto libre

### Actualizar Muestra

```http
PUT /api/v1/samples/{sample_id}
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "name": "Muestra_001_Actualizada",
  "description": "Descripción actualizada",
  "metadata": {
    "notes": "Notas adicionales del análisis"
  }
}
```

### Eliminar Muestra

```http
DELETE /api/v1/samples/{sample_id}
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "message": "Muestra eliminada exitosamente",
  "deleted_at": "2024-01-15T15:00:00Z"
}
```

### Búsqueda Avanzada

```http
POST /api/v1/samples/search
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "filters": {
    "type": ["casquillo", "proyectil"],
    "status": ["processed"],
    "date_range": {
      "from": "2024-01-01",
      "to": "2024-01-31"
    },
    "case_ids": ["CASO_2024_001", "CASO_2024_002"]
  },
  "sort": {
    "field": "created_at",
    "order": "desc"
  },
  "pagination": {
    "page": 1,
    "limit": 50
  }
}
```

---

## Endpoints de Comparación

### Comparar Dos Muestras

```http
POST /api/v1/comparisons/compare
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "sample_a": "BAL_2024_001",
  "sample_b": "BAL_2024_002",
  "metric": "cosine_similarity",
  "parameters": {
    "threshold": 0.8,
    "normalize": true
  }
}
```

**Respuesta**:
```json
{
  "id": "comparison_67890",
  "sample_a": "BAL_2024_001",
  "sample_b": "BAL_2024_002",
  "metric": "cosine_similarity",
  "similarity": 0.87,
  "confidence": 0.94,
  "classification": "high",
  "details": {
    "matching_features": 156,
    "total_features": 256,
    "match_percentage": 60.9,
    "statistical_significance": 0.001
  },
  "processing_time": 0.15,
  "created_at": "2024-01-15T15:30:00Z"
}
```

### Comparación Masiva

```http
POST /api/v1/comparisons/batch
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "samples": ["BAL_2024_001", "BAL_2024_002", "BAL_2024_003"],
  "metric": "cosine_similarity",
  "parameters": {
    "threshold": 0.7,
    "max_comparisons": 1000
  }
}
```

**Respuesta**:
```json
{
  "id": "batch_comparison_123",
  "status": "processing",
  "total_comparisons": 3,
  "completed_comparisons": 0,
  "estimated_completion": "2024-01-15T15:35:00Z",
  "results_url": "/api/v1/comparisons/batch/batch_comparison_123/results"
}
```

### Obtener Resultados de Comparación Masiva

```http
GET /api/v1/comparisons/batch/{batch_id}/results
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "id": "batch_comparison_123",
  "status": "completed",
  "matrix": [
    {
      "sample_a": "BAL_2024_001",
      "sample_b": "BAL_2024_002",
      "similarity": 0.23,
      "classification": "low"
    },
    {
      "sample_a": "BAL_2024_001",
      "sample_b": "BAL_2024_003",
      "similarity": 0.87,
      "classification": "high"
    },
    {
      "sample_a": "BAL_2024_002",
      "sample_b": "BAL_2024_003",
      "similarity": 0.34,
      "classification": "medium"
    }
  ],
  "statistics": {
    "mean_similarity": 0.48,
    "std_similarity": 0.32,
    "high_matches": 1,
    "medium_matches": 1,
    "low_matches": 1
  }
}
```

### Buscar Muestras Similares

```http
POST /api/v1/comparisons/search-similar
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "reference_sample": "BAL_2024_001",
  "threshold": 0.8,
  "limit": 10,
  "filters": {
    "type": "casquillo",
    "exclude_cases": ["CASO_2024_001"]
  }
}
```

**Respuesta**:
```json
{
  "reference_sample": "BAL_2024_001",
  "matches": [
    {
      "sample_id": "BAL_2024_003",
      "similarity": 0.87,
      "confidence": 0.94,
      "classification": "high"
    },
    {
      "sample_id": "BAL_2024_015",
      "similarity": 0.82,
      "confidence": 0.89,
      "classification": "high"
    }
  ],
  "total_matches": 2,
  "search_time": 1.23
}
```

---

## Endpoints de Configuración

### Obtener Configuración

```http
GET /api/v1/config
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "algorithms": {
    "lbp": {
      "radius": 2,
      "points": 16,
      "uniform": true
    },
    "sift": {
      "n_features": 500,
      "contrast_threshold": 0.04,
      "edge_threshold": 10
    }
  },
  "processing": {
    "max_image_size": 52428800,
    "timeout": 300,
    "parallel_processing": true
  },
  "comparison": {
    "default_metric": "cosine_similarity",
    "default_threshold": 0.8
  }
}
```

### Actualizar Configuración

```http
PUT /api/v1/config
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "algorithms": {
    "lbp": {
      "radius": 3,
      "points": 24
    }
  },
  "processing": {
    "timeout": 600
  }
}
```

### Obtener Información del Sistema

```http
GET /api/v1/system/info
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "version": "1.0.0",
  "build": "2024.01.15.001",
  "environment": "production",
  "uptime": 86400,
  "database": {
    "status": "connected",
    "version": "PostgreSQL 13.7"
  },
  "features": {
    "algorithms": ["lbp", "sift", "orb"],
    "metrics": ["cosine_similarity", "euclidean_distance", "pearson_correlation"],
    "formats": ["jpg", "png", "tiff", "bmp"]
  }
}
```

---

## Endpoints de Monitoreo

### Métricas del Sistema

```http
GET /api/v1/monitoring/metrics
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "timestamp": "2024-01-15T16:00:00Z",
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1,
    "load_average": [1.2, 1.5, 1.8]
  },
  "application": {
    "active_analyses": 5,
    "queue_size": 12,
    "total_samples": 1250,
    "daily_analyses": 87
  },
  "database": {
    "connections": 15,
    "query_time_avg": 0.045,
    "cache_hit_rate": 0.92
  }
}
```

### Estado de Salud

```http
GET /api/v1/health
```

**Respuesta**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T16:00:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 0.023
    },
    "storage": {
      "status": "healthy",
      "available_space": "85%"
    },
    "algorithms": {
      "status": "healthy",
      "available": ["lbp", "sift", "orb"]
    }
  },
  "version": "1.0.0"
}
```

### Alertas

```http
GET /api/v1/monitoring/alerts
Authorization: Bearer YOUR_TOKEN
```

**Respuesta**:
```json
{
  "alerts": [
    {
      "id": "alert_001",
      "level": "warning",
      "message": "Uso de memoria alto: 85%",
      "timestamp": "2024-01-15T15:45:00Z",
      "resolved": false
    },
    {
      "id": "alert_002",
      "level": "info",
      "message": "Análisis completado exitosamente",
      "timestamp": "2024-01-15T15:30:00Z",
      "resolved": true
    }
  ],
  "total": 2,
  "unresolved": 1
}
```

---

## Modelos de Datos

### Sample (Muestra)

```json
{
  "id": "string",
  "name": "string",
  "type": "casquillo|proyectil|arma",
  "status": "pending|processing|processed|failed",
  "description": "string",
  "image_url": "string",
  "features": {
    "algorithm": "string",
    "dimensions": "integer",
    "data": "array",
    "extracted_at": "datetime"
  },
  "metadata": {
    "case_id": "string",
    "investigator": "string",
    "collection_date": "date",
    "location": "string",
    "notes": "string"
  },
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Analysis (Análisis)

```json
{
  "id": "string",
  "sample_id": "string",
  "algorithm": "lbp|sift|orb",
  "status": "pending|processing|completed|failed",
  "progress": "integer",
  "parameters": "object",
  "features": {
    "dimensions": "integer",
    "histogram": "array",
    "statistics": "object"
  },
  "processing_time": "float",
  "error_message": "string",
  "created_at": "datetime",
  "completed_at": "datetime"
}
```

### Comparison (Comparación)

```json
{
  "id": "string",
  "sample_a": "string",
  "sample_b": "string",
  "metric": "string",
  "similarity": "float",
  "confidence": "float",
  "classification": "high|medium|low",
  "details": {
    "matching_features": "integer",
    "total_features": "integer",
    "match_percentage": "float",
    "statistical_significance": "float"
  },
  "processing_time": "float",
  "created_at": "datetime"
}
```

### User (Usuario)

```json
{
  "id": "string",
  "username": "string",
  "email": "string",
  "role": "admin|analyst|viewer",
  "permissions": "array",
  "last_login": "datetime",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

---

## Códigos de Error

### Códigos HTTP Estándar

| Código | Descripción | Uso |
|--------|-------------|-----|
| 200 | OK | Petición exitosa |
| 201 | Created | Recurso creado |
| 400 | Bad Request | Petición inválida |
| 401 | Unauthorized | No autenticado |
| 403 | Forbidden | Sin permisos |
| 404 | Not Found | Recurso no encontrado |
---

## Error Handling

All API methods follow consistent error handling patterns:

```python
try:
    result = pipeline.process_comparison(image1, image2)
    if result.error_messages:
        print("Warnings:", result.warnings)
        print("Errors:", result.error_messages)
except Exception as e:
    logger.error(f"Processing failed: {e}", exc_info=True)
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 400 | Bad Request | Invalid parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Invalid data |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

---

## Configuration Examples

### Basic Configuration

```yaml
processing:
  level: "standard"
  enable_quality_assessment: true
  enable_preprocessing: true
  enable_roi_detection: true
  enable_matching: true
  enable_cmc_analysis: true

algorithms:
  feature_extraction: "orb"
  matching_algorithm: "brute_force"
  
quality:
  min_quality_score: 0.5
  snr_threshold: 10.0
```

### Advanced Configuration

```yaml
processing:
  level: "forensic"
  parallel_processing: true
  cache_enabled: true
  export_intermediate_results: true

hardware:
  gpu_enabled: true
  gpu_memory_limit: "4GB"
  cpu_threads: 8

nist_compliance:
  strict_validation: true
  quality_standards: "NIST_SP_800_101"
  
afte_conclusion:
  identification_threshold: 0.8
  confidence_threshold: 0.9
  min_cmc_count: 10
```

---

## Version Information

- **API Version**: 1.0.0
- **Compatible Python**: 3.8+
- **Required Dependencies**: See `requirements.txt`
- **Optional Dependencies**: GPU acceleration, advanced visualization

## Support

For technical support and documentation updates, please refer to the project repository and issue tracker.

---

## Legacy REST API Documentation

### Error Codes (Legacy)

```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "El formato de imagen no es soportado",
    "details": {
      "supported_formats": ["jpg", "png", "tiff", "bmp"],
      "received_format": "gif"
    },
    "timestamp": "2024-01-15T16:00:00Z"
  }
}
```

#### Errores de Análisis

| Código | Descripción |
|--------|-------------|
| `INVALID_IMAGE_FORMAT` | Formato de imagen no soportado |
| `IMAGE_TOO_LARGE` | Imagen excede tamaño máximo |
| `ALGORITHM_NOT_AVAILABLE` | Algoritmo no disponible |
| `PROCESSING_TIMEOUT` | Tiempo de procesamiento excedido |
| `INSUFFICIENT_FEATURES` | Características insuficientes extraídas |

#### Errores de Base de Datos

| Código | Descripción |
|--------|-------------|
| `SAMPLE_NOT_FOUND` | Muestra no encontrada |
| `DUPLICATE_SAMPLE_ID` | ID de muestra duplicado |
| `DATABASE_CONNECTION_ERROR` | Error de conexión a base de datos |
| `INVALID_QUERY_PARAMETERS` | Parámetros de consulta inválidos |

#### Errores de Comparación

| Código | Descripción |
|--------|-------------|
| `SAMPLES_NOT_COMPATIBLE` | Muestras no compatibles para comparación |
| `MISSING_FEATURES` | Características faltantes en las muestras |
| `INVALID_SIMILARITY_METRIC` | Métrica de similitud inválida |
| `COMPARISON_TIMEOUT` | Tiempo de comparación excedido |

---

## Ejemplos de Uso

### Ejemplo 1: Análisis Completo de Imagen

```python
import requests
import json

# Configuración
API_BASE = "https://api.sigec-balistica.com/v1"
API_KEY = "your_api_key_here"
headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Crear muestra
sample_data = {
    "name": "Casquillo_Escena_A",
    "type": "casquillo",
    "description": "Casquillo encontrado en escena del crimen",
    "metadata": {
        "case_id": "CASO_2024_001",
        "investigator": "Detective Smith",
        "collection_date": "2024-01-15"
    }
}

response = requests.post(
    f"{API_BASE}/samples",
    headers=headers,
    json=sample_data
)
sample = response.json()
sample_id = sample["id"]

# 2. Procesar imagen
with open("casquillo.jpg", "rb") as image_file:
    files = {"image": image_file}
    data = {
        "algorithm": "lbp",
        "parameters": json.dumps({
            "radius": 2,
            "points": 16
        })
    }
    
    response = requests.post(
        f"{API_BASE}/analysis/process",
        headers=headers,
        files=files,
        data=data
    )
    analysis = response.json()
    analysis_id = analysis["id"]

# 3. Verificar estado del análisis
while True:
    response = requests.get(
        f"{API_BASE}/analysis/{analysis_id}",
        headers=headers
    )
    status = response.json()
    
    if status["status"] == "completed":
        print("Análisis completado exitosamente")
        break
    elif status["status"] == "failed":
        print("Error en el análisis")
        break
    else:
        print(f"Progreso: {status['progress']}%")
        time.sleep(5)

# 4. Obtener resultados
response = requests.get(
    f"{API_BASE}/samples/{sample_id}",
    headers=headers
)
sample_with_features = response.json()
print(f"Características extraídas: {sample_with_features['features']['dimensions']}")
```

### Ejemplo 2: Comparación de Muestras

```python
# Comparar dos muestras
comparison_data = {
    "sample_a": "BAL_2024_001",
    "sample_b": "BAL_2024_002",
    "metric": "cosine_similarity",
    "parameters": {
        "threshold": 0.8
    }
}

response = requests.post(
    f"{API_BASE}/comparisons/compare",
    headers=headers,
    json=comparison_data
)
comparison = response.json()

print(f"Similitud: {comparison['similarity']:.2f}")
print(f"Clasificación: {comparison['classification']}")
print(f"Confianza: {comparison['confidence']:.2f}")
```

### Ejemplo 3: Búsqueda de Muestras Similares

```python
# Buscar muestras similares
search_data = {
    "reference_sample": "BAL_2024_001",
    "threshold": 0.7,
    "limit": 5,
    "filters": {
        "type": "casquillo"
    }
}

response = requests.post(
    f"{API_BASE}/comparisons/search-similar",
    headers=headers,
    json=search_data
)
results = response.json()

print(f"Encontradas {len(results['matches'])} muestras similares:")
for match in results["matches"]:
    print(f"- {match['sample_id']}: {match['similarity']:.2f}")
```

### Ejemplo 4: Monitoreo del Sistema

```python
# Obtener métricas del sistema
response = requests.get(
    f"{API_BASE}/monitoring/metrics",
    headers=headers
)
metrics = response.json()

print(f"CPU: {metrics['system']['cpu_usage']}%")
print(f"Memoria: {metrics['system']['memory_usage']}%")
print(f"Análisis activos: {metrics['application']['active_analyses']}")
```

---

## SDKs y Librerías

### Python SDK

```bash
pip install sigec-balistica-sdk
```

```python
from sigec_balistica import SIGeCClient

# Inicializar cliente
client = SIGeCClient(
    api_key="your_api_key",
    base_url="https://api.sigec-balistica.com/v1"
)

# Analizar imagen
analysis = client.analyze_image(
    image_path="casquillo.jpg",
    algorithm="lbp",
    metadata={
        "case_id": "CASO_2024_001",
        "investigator": "Detective Smith"
    }
)

# Comparar muestras
comparison = client.compare_samples(
    sample_a="BAL_2024_001",
    sample_b="BAL_2024_002",
    metric="cosine_similarity"
)

print(f"Similitud: {comparison.similarity}")
```

### JavaScript SDK

```bash
npm install sigec-balistica-sdk
```

```javascript
const { SIGeCClient } = require('sigec-balistica-sdk');

// Inicializar cliente
const client = new SIGeCClient({
    apiKey: 'your_api_key',
    baseUrl: 'https://api.sigec-balistica.com/v1'
});

// Analizar imagen
const analysis = await client.analyzeImage({
    imagePath: 'casquillo.jpg',
    algorithm: 'lbp',
    metadata: {
        caseId: 'CASO_2024_001',
        investigator: 'Detective Smith'
    }
});

// Comparar muestras
const comparison = await client.compareSamples({
    sampleA: 'BAL_2024_001',
    sampleB: 'BAL_2024_002',
    metric: 'cosine_similarity'
});

console.log(`Similitud: ${comparison.similarity}`);
```

### Java SDK

```xml
<dependency>
    <groupId>com.sigec</groupId>
    <artifactId>sigec-balistica-sdk</artifactId>
    <version>1.0.0</version>
</dependency>
```

```java
import com.sigec.balistica.SIGeCClient;
import com.sigec.balistica.models.*;

// Inicializar cliente
SIGeCClient client = new SIGeCClient.Builder()
    .apiKey("your_api_key")
    .baseUrl("https://api.sigec-balistica.com/v1")
    .build();

// Analizar imagen
AnalysisRequest request = new AnalysisRequest.Builder()
    .imagePath("casquillo.jpg")
    .algorithm("lbp")
    .addMetadata("case_id", "CASO_2024_001")
    .addMetadata("investigator", "Detective Smith")
    .build();

Analysis analysis = client.analyzeImage(request);

// Comparar muestras
ComparisonRequest compRequest = new ComparisonRequest.Builder()
    .sampleA("BAL_2024_001")
    .sampleB("BAL_2024_002")
    .metric("cosine_similarity")
    .build();

Comparison comparison = client.compareSamples(compRequest);
System.out.println("Similitud: " + comparison.getSimilarity());
```

### C# SDK

```bash
dotnet add package SIGeC.Balistica.SDK
```

```csharp
using SIGeC.Balistica;

// Inicializar cliente
var client = new SIGeCClient(new SIGeCClientOptions
{
    ApiKey = "your_api_key",
    BaseUrl = "https://api.sigec-balistica.com/v1"
});

// Analizar imagen
var analysis = await client.AnalyzeImageAsync(new AnalysisRequest
{
    ImagePath = "casquillo.jpg",
    Algorithm = "lbp",
    Metadata = new Dictionary<string, string>
    {
        ["case_id"] = "CASO_2024_001",
        ["investigator"] = "Detective Smith"
    }
});

// Comparar muestras
var comparison = await client.CompareSamplesAsync(new ComparisonRequest
{
    SampleA = "BAL_2024_001",
    SampleB = "BAL_2024_002",
    Metric = "cosine_similarity"
});

Console.WriteLine($"Similitud: {comparison.Similarity}");
```

---

## Webhooks

### Configurar Webhooks

```http
POST /api/v1/webhooks
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/sigec",
  "events": ["analysis.completed", "comparison.completed"],
  "secret": "your_webhook_secret"
}
```

### Eventos Disponibles

| Evento | Descripción |
|--------|-------------|
| `analysis.started` | Análisis iniciado |
| `analysis.completed` | Análisis completado |
| `analysis.failed` | Análisis falló |
| `comparison.completed` | Comparación completada |
| `sample.created` | Muestra creada |
| `sample.updated` | Muestra actualizada |
| `alert.triggered` | Alerta activada |

### Ejemplo de Payload

```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T16:00:00Z",
  "data": {
    "analysis_id": "analysis_12345",
    "sample_id": "BAL_2024_001",
    "status": "completed",
    "processing_time": 2.3,
    "features": {
      "dimensions": 256,
      "algorithm": "lbp"
    }
  }
}
```

---

## Rate Limiting

### Límites por Endpoint

| Endpoint | Límite | Ventana |
|----------|--------|---------|
| `/analysis/process` | 100/hora | 1 hora |
| `/comparisons/compare` | 1000/hora | 1 hora |
| `/samples/*` | 5000/hora | 1 hora |
| `/monitoring/*` | 10000/hora | 1 hora |

### Headers de Rate Limiting

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
```

### Manejo de Rate Limiting

```python
import time
import requests

def make_request_with_retry(url, headers, data=None, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 429:
            # Rate limit exceeded
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            wait_time = reset_time - int(time.time())
            
            if wait_time > 0:
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
        
        return response
    
    raise Exception("Max retries exceeded")
```

---

## Versionado de API

### Estrategia de Versionado

- **URL Path**: `/api/v1/`, `/api/v2/`
- **Backward Compatibility**: Mantenida por 2 versiones
- **Deprecation**: Notificación 6 meses antes
- **Migration Guide**: Documentación de migración

### Versiones Disponibles

| Versión | Estado | Soporte hasta |
|---------|--------|---------------|
| v1 | Actual | 2025-12-31 |
| v2 | Beta | - |

### Headers de Versión

```http
API-Version: 1.0
Deprecated-Version: false
Sunset: 2025-12-31T23:59:59Z
```

---

*Documentación de API - SIGeC-Balistica v1.0*  
*Última actualización: Enero 2024*  
*© 2024 SIGeC-Balistica. Todos los derechos reservados.*