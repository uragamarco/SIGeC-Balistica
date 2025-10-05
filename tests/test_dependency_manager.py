#!/usr/bin/env python3
"""
Tests para el Sistema de Gestión de Dependencias de SEACABAr
===========================================================

Autor: SEACABAr Team
Versión: 1.0.0
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Importar módulos a testear
from utils.dependency_manager import (
    DependencyManager, DependencyInfo, DependencyType, DependencyStatus,
    dependency_manager, check_dependencies, safe_import, install_missing,
    generate_dependency_report
)
from utils.fallback_implementations import (
    DeepLearningFallback, WebServiceFallback, ImageProcessingFallback,
    DatabaseFallback, FallbackRegistry, get_fallback
)


class TestDependencyInfo:
    """Tests para la clase DependencyInfo"""
    
    def test_dependency_info_creation(self):
        """Test creación de DependencyInfo"""
        dep = DependencyInfo(
            name="test-package",
            import_name="test_package",
            version_required="1.0.0",
            dependency_type=DependencyType.REQUIRED
        )
        
        assert dep.name == "test-package"
        assert dep.import_name == "test_package"
        assert dep.version_required == "1.0.0"
        assert dep.dependency_type == DependencyType.REQUIRED
        assert dep.status == DependencyStatus.MISSING
        assert not dep.fallback_available
    
    def test_dependency_info_defaults(self):
        """Test valores por defecto de DependencyInfo"""
        dep = DependencyInfo(name="test", import_name="test")
        
        assert dep.version_required is None
        assert dep.version_installed is None
        assert dep.dependency_type == DependencyType.REQUIRED
        assert not dep.fallback_available
        assert dep.fallback_function is None
        assert dep.status == DependencyStatus.MISSING
        assert dep.error_message is None


class TestDependencyManager:
    """Tests para la clase DependencyManager"""
    
    def setup_method(self):
        """Configuración para cada test"""
        self.manager = DependencyManager()
    
    def test_initialization(self):
        """Test inicialización del gestor"""
        assert isinstance(self.manager.dependencies, dict)
        assert len(self.manager.dependencies) > 0
        assert isinstance(self.manager.fallbacks, dict)
    
    def test_check_existing_dependency(self):
        """Test verificación de dependencia existente"""
        # numpy debería estar disponible en el entorno de testing
        result = self.manager.check_dependency("numpy")
        
        assert isinstance(result, DependencyInfo)
        assert result.name == "numpy"
        assert result.status in [DependencyStatus.AVAILABLE, DependencyStatus.VERSION_MISMATCH]
    
    def test_check_nonexistent_dependency(self):
        """Test verificación de dependencia inexistente"""
        result = self.manager.check_dependency("nonexistent-package-xyz")
        
        assert result.status == DependencyStatus.MISSING
        assert "no registrada" in result.error_message
    
    @patch('importlib.import_module')
    def test_check_dependency_import_error(self, mock_import):
        """Test manejo de errores de importación"""
        mock_import.side_effect = ImportError("Test import error")
        
        result = self.manager.check_dependency("numpy")
        
        assert result.status == DependencyStatus.IMPORT_ERROR
        assert "Test import error" in result.error_message
    
    def test_check_all_dependencies(self):
        """Test verificación de todas las dependencias"""
        results = self.manager.check_all_dependencies()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for package_name, dep_info in results.items():
            assert isinstance(dep_info, DependencyInfo)
            assert dep_info.name == package_name
    
    def test_get_missing_dependencies(self):
        """Test obtención de dependencias faltantes"""
        missing = self.manager.get_missing_dependencies()
        
        assert isinstance(missing, list)
        # Puede estar vacía si todas las dependencias están instaladas
    
    def test_get_missing_dependencies_include_optional(self):
        """Test obtención incluyendo dependencias opcionales"""
        missing_all = self.manager.get_missing_dependencies(include_optional=True)
        missing_required = self.manager.get_missing_dependencies(include_optional=False)
        
        assert isinstance(missing_all, list)
        assert isinstance(missing_required, list)
        assert len(missing_all) >= len(missing_required)
    
    @patch('subprocess.run')
    def test_install_dependency_success(self, mock_run):
        """Test instalación exitosa de dependencia"""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")
        
        result = self.manager.install_dependency("test-package", "1.0.0")
        
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "test-package==1.0.0" in args
    
    @patch('subprocess.run')
    def test_install_dependency_failure(self, mock_run):
        """Test fallo en instalación de dependencia"""
        mock_run.side_effect = subprocess.CalledProcessError(1, "pip", stderr="Error")
        
        result = self.manager.install_dependency("nonexistent-package")
        
        assert result is False
    
    def test_safe_import_available_dependency(self):
        """Test importación segura de dependencia disponible"""
        # Usar numpy que debería estar disponible y registrado
        result = self.manager.safe_import("numpy")
        
        assert result is not None
    
    def test_safe_import_missing_required(self):
        """Test importación segura de dependencia requerida faltante"""
        # Crear dependencia requerida falsa
        self.manager.dependencies["fake-required"] = DependencyInfo(
            name="fake-required",
            import_name="fake_required",
            dependency_type=DependencyType.REQUIRED
        )
        
        with pytest.raises(ImportError):
            self.manager.safe_import("fake-required")
    
    def test_safe_import_missing_optional_with_fallback(self):
        """Test importación segura de dependencia opcional con fallback"""
        # Crear dependencia opcional con fallback
        self.manager.dependencies["fake-optional"] = DependencyInfo(
            name="fake-optional",
            import_name="fake_optional",
            dependency_type=DependencyType.OPTIONAL,
            fallback_available=True
        )
        
        # Mock del fallback
        mock_fallback = Mock()
        self.manager.fallbacks["fake-optional"] = mock_fallback
        
        result = self.manager.safe_import("fake-optional")
        
        assert result == mock_fallback
    
    def test_generate_report(self):
        """Test generación de reporte"""
        report = self.manager.generate_report()
        
        assert isinstance(report, dict)
        assert "timestamp" in report
        assert "python_version" in report
        assert "total_dependencies" in report
        assert "by_status" in report
        assert "by_type" in report
        assert "details" in report
        
        # Verificar estructura de contadores
        assert isinstance(report["by_status"], dict)
        assert isinstance(report["by_type"], dict)
        assert isinstance(report["details"], dict)


class TestFallbackImplementations:
    """Tests para las implementaciones de fallback"""
    
    def test_deep_learning_fallback(self):
        """Test fallback de deep learning"""
        fallback = DeepLearningFallback()
        
        assert not fallback.available
        assert fallback.backend == "numpy"
        
        # Test extracción de características
        import numpy as np
        test_image = np.random.rand(50, 50)
        features = fallback.extract_features(test_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_web_service_fallback(self):
        """Test fallback de servicios web"""
        fallback = WebServiceFallback()
        
        assert not fallback.available
        assert fallback.server is None
    
    def test_image_processing_fallback(self):
        """Test fallback de procesamiento de imágenes"""
        fallback = ImageProcessingFallback()
        
        assert not fallback.available
        
        # Test reducción de ruido
        import numpy as np
        test_image = np.random.rand(20, 20)
        processed = fallback.advanced_noise_reduction(test_image)
        
        assert isinstance(processed, np.ndarray)
        assert processed.shape == test_image.shape
    
    def test_database_fallback(self):
        """Test fallback de base de datos"""
        fallback = DatabaseFallback()
        
        assert not fallback.available
        
        # Test índice vectorial simple
        index = fallback.create_vector_index(10)
        
        import numpy as np
        vectors = np.random.rand(5, 10)
        index.add(vectors)
        
        query = np.random.rand(10)
        distances, indices = index.search(query, k=3)
        
        assert len(distances) == 1
        assert len(indices) == 1
        assert len(distances[0]) <= 3
        assert len(indices[0]) <= 3


class TestFallbackRegistry:
    """Tests para el registro de fallbacks"""
    
    def setup_method(self):
        """Configuración para cada test"""
        self.registry = FallbackRegistry()
    
    def test_initialization(self):
        """Test inicialización del registro"""
        assert isinstance(self.registry.fallbacks, dict)
        assert len(self.registry.fallbacks) > 0
    
    def test_get_fallback(self):
        """Test obtención de fallback"""
        fallback = self.registry.get_fallback('deep_learning')
        
        assert fallback is not None
        assert isinstance(fallback, DeepLearningFallback)
    
    def test_register_fallback(self):
        """Test registro de nuevo fallback"""
        mock_fallback = Mock()
        self.registry.register_fallback('test_category', mock_fallback)
        
        retrieved = self.registry.get_fallback('test_category')
        assert retrieved == mock_fallback
    
    def test_list_available_fallbacks(self):
        """Test listado de fallbacks disponibles"""
        available = self.registry.list_available_fallbacks()
        
        assert isinstance(available, list)
        assert len(available) > 0
        assert 'deep_learning' in available


class TestGlobalFunctions:
    """Tests para las funciones globales"""
    
    def test_check_dependencies(self):
        """Test función global check_dependencies"""
        result = check_dependencies()
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_generate_dependency_report(self):
        """Test función global generate_dependency_report"""
        report = generate_dependency_report()
        
        assert isinstance(report, dict)
        assert "total_dependencies" in report
    
    def test_get_fallback(self):
        """Test función global get_fallback"""
        fallback = get_fallback('deep_learning')
        
        assert fallback is not None
        assert isinstance(fallback, DeepLearningFallback)


class TestIntegration:
    """Tests de integración del sistema completo"""
    
    def test_full_dependency_check_workflow(self):
        """Test flujo completo de verificación de dependencias"""
        # 1. Verificar dependencias
        results = check_dependencies()
        assert isinstance(results, dict)
        
        # 2. Generar reporte
        report = generate_dependency_report()
        assert report["total_dependencies"] == len(results)
        
        # 3. Verificar consistencia
        for package_name, dep_info in results.items():
            assert package_name in report["details"]
            detail = report["details"][package_name]
            assert detail["status"] == dep_info.status.value
    
    def test_fallback_integration(self):
        """Test integración de fallbacks"""
        # Verificar que los fallbacks están disponibles
        categories = ['deep_learning', 'web_service', 'image_processing', 'database']
        
        for category in categories:
            fallback = get_fallback(category)
            assert fallback is not None
            assert not fallback.available  # Los fallbacks no son la implementación principal
    
    def test_safe_import_integration(self):
        """Test integración de importación segura"""
        # Test con una dependencia que existe
        result = safe_import("numpy")
        
        assert result is not None


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"])