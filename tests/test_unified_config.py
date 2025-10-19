#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruebas para el Sistema de Configuración Unificado - SIGeC-Balistica
============================================================

Suite completa de pruebas para validar el funcionamiento del sistema
de configuración unificado, incluyendo validación, migración, y
compatibilidad con configuraciones legacy.

Autor: SIGeC-BalisticaTeam
Fecha: Octubre 2025
"""

import os
import sys
import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# Agregar el directorio raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.unified_config import (
    UnifiedConfig,
    DatabaseConfig,
    ImageProcessingConfig,
    MatchingConfig,
    GUIConfig,
    LoggingConfig,
    DeepLearningConfig,
    NISTConfig,
    Environment,
    ConfigValidationError,
    ConfigMigrationError,
    get_unified_config,
    reset_unified_config,
    get_database_config,
    get_image_processing_config,
    get_matching_config,
    get_gui_config,
    get_logging_config,
    get_deep_learning_config,
    get_nist_config
)
from matching.unified_matcher import AlgorithmType, MatchingLevel

class TestDatabaseConfig:
    """Pruebas para DatabaseConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = DatabaseConfig()
        
        assert config.sqlite_path == "database/ballistics.db"
        assert config.faiss_index_path == "database/faiss_index"
        assert config.backup_enabled is True
        assert config.backup_interval_hours == 24
        assert config.connection_pool_size == 10
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = DatabaseConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = DatabaseConfig(
            backup_interval_hours=0,
            backup_retention_days=0,
            connection_pool_size=0,
            query_timeout=0
        )
        
        errors = config.validate()
        assert len(errors) == 4
        assert any("backup_interval_hours" in error for error in errors)
        assert any("backup_retention_days" in error for error in errors)
        assert any("connection_pool_size" in error for error in errors)
        assert any("query_timeout" in error for error in errors)

class TestImageProcessingConfig:
    """Pruebas para ImageProcessingConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = ImageProcessingConfig()
        
        assert config.max_image_size == 2048
        assert config.min_image_size == 64
        assert config.orb_features == 5000
        assert config.roi_detection_method == "simple"
        assert config.enable_parallel_processing is True
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = ImageProcessingConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = ImageProcessingConfig(
            max_image_size=32,  # Menor que min_image_size
            min_image_size=64,
            orb_features=50,  # Muy bajo
            roi_detection_method="invalid",
            gaussian_kernel_size=4,  # Par
            clahe_clip_limit=15.0,  # Muy alto
            max_workers=0
        )
        
        errors = config.validate()
        assert len(errors) >= 5
        assert any("max_image_size debe ser mayor que min_image_size" in error for error in errors)
        assert any("orb_features debe ser al menos 100" in error for error in errors)
        assert any("roi_detection_method debe ser" in error for error in errors)
        assert any("gaussian_kernel_size debe ser impar" in error for error in errors)
        assert any("max_workers debe ser mayor a 0" in error for error in errors)

class TestMatchingConfig:
    """Pruebas para MatchingConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = MatchingConfig()
        
        assert config.matcher_type == "BF"
        assert config.distance_threshold == 0.75
        assert config.min_matches == 10
        assert config.enable_geometric_verification is True
        assert config.enable_firing_pin_analysis is True
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = MatchingConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = MatchingConfig(
            matcher_type="INVALID",
            distance_threshold=1.5,  # Muy alto
            min_matches=2,  # Muy bajo
            similarity_threshold=1.5,  # Muy alto
            ransac_threshold=100.0,  # Muy alto
            batch_size=0
        )
        
        errors = config.validate()
        assert len(errors) >= 5
        assert any("matcher_type debe ser" in error for error in errors)
        assert any("distance_threshold debe estar entre" in error for error in errors)
        assert any("min_matches debe ser al menos 4" in error for error in errors)
        assert any("batch_size debe ser mayor a 0" in error for error in errors)

class TestGUIConfig:
    """Pruebas para GUIConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = GUIConfig()
        
        assert config.window_width == 1200
        assert config.window_height == 800
        assert config.theme == "default"
        assert config.language == "es"
        assert config.auto_save is True
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = GUIConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = GUIConfig(
            window_width=500,  # Muy pequeño
            window_height=400,  # Muy pequeño
            theme="invalid",
            language="invalid",
            font_size=30,  # Muy grande
            auto_save_interval=30,  # Muy bajo
            max_zoom_level=0.05,  # Menor que min_zoom_level
            default_image_quality=30  # Muy bajo
        )
        
        errors = config.validate()
        assert len(errors) >= 6
        assert any("window_width debe ser al menos" in error for error in errors)
        assert any("theme debe ser" in error for error in errors)
        assert any("language debe ser" in error for error in errors)
        assert any("font_size debe estar entre" in error for error in errors)

class TestLoggingConfig:
    """Pruebas para LoggingConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.console_output is True
        assert config.file_path == "logs/ballistics.log"
        assert config.enable_rotation is True
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = LoggingConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = LoggingConfig(
            level="INVALID",
            backup_count=-1,
            max_file_size="invalid_size"
        )
        
        errors = config.validate()
        assert len(errors) >= 2
        assert any("level debe ser uno de" in error for error in errors)
        assert any("backup_count debe ser mayor o igual a 0" in error for error in errors)

class TestDeepLearningConfig:
    """Pruebas para DeepLearningConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = DeepLearningConfig()
        
        assert config.enabled is False
        assert config.device == "cpu"
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.epochs == 100
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = DeepLearningConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = DeepLearningConfig(
            device="invalid",
            batch_size=0,
            learning_rate=2.0,  # Muy alto
            epochs=0,
            dropout_rate=1.5  # Muy alto
        )
        
        errors = config.validate()
        assert len(errors) >= 4
        assert any("device debe ser" in error for error in errors)
        assert any("batch_size debe ser mayor a 0" in error for error in errors)
        assert any("learning_rate debe estar entre" in error for error in errors)
        assert any("epochs debe ser mayor a 0" in error for error in errors)

class TestNISTConfig:
    """Pruebas para NISTConfig"""
    
    def test_default_values(self):
        """Prueba valores por defecto"""
        config = NISTConfig()
        
        assert config.enable_nist_compliance is True
        assert config.enable_afte_standards is True
        assert config.min_quality_score == 0.7
        assert config.report_format == "json"
    
    def test_validation_success(self):
        """Prueba validación exitosa"""
        config = NISTConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_validation_errors(self):
        """Prueba errores de validación"""
        config = NISTConfig(
            min_quality_score=1.5,  # Muy alto
            quality_threshold=1.5,  # Muy alto
            report_format="invalid"
        )
        
        errors = config.validate()
        assert len(errors) >= 3
        assert any("min_quality_score debe estar entre" in error for error in errors)
        assert any("quality_threshold debe estar entre" in error for error in errors)
        assert any("report_format debe ser" in error for error in errors)

class TestUnifiedConfig:
    """Pruebas para UnifiedConfig"""
    
    def setup_method(self):
        """Configuración para cada prueba"""
        # Resetear instancia global
        reset_unified_config()
        
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        reset_unified_config()
        
        # Limpiar directorio temporal
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
    
    @patch('config.unified_config.Path.cwd')
    def test_initialization_default(self, mock_cwd):
        """Prueba inicialización con valores por defecto"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.project_root == self.temp_path
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.image_processing, ImageProcessingConfig)
        assert isinstance(config.matching, MatchingConfig)
        assert isinstance(config.gui, GUIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.deep_learning, DeepLearningConfig)
        assert isinstance(config.nist, NISTConfig)
    
    @patch('config.unified_config.Path.cwd')
    def test_environment_detection(self, mock_cwd):
        """Prueba detección de entorno"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        # Probar entorno de producción
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'production'}):
            config = UnifiedConfig()
            assert config.environment == Environment.PRODUCTION
        
        # Probar entorno de testing
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'testing'}):
            config = UnifiedConfig()
            assert config.environment == Environment.TESTING
    
    @patch('config.unified_config.Path.cwd')
    def test_save_and_load_config(self, mock_cwd):
        """Prueba guardar y cargar configuración"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        # Crear configuración
        config = UnifiedConfig()
        
        # Modificar algunos valores
        config.database.sqlite_path = "custom/path.db"
        config.gui.theme = "dark"
        config.matching.min_matches = 15
        
        # Guardar configuración
        config.save_config()
        
        # Verificar que el archivo existe
        assert config.config_path.exists()
        
        # Crear nueva instancia y verificar que carga los valores
        config2 = UnifiedConfig()
        assert config2.database.sqlite_path == "custom/path.db"
        assert config2.gui.theme == "dark"
        assert config2.matching.min_matches == 15
    
    @patch('config.unified_config.Path.cwd')
    def test_validation_all_sections(self, mock_cwd):
        """Prueba validación de todas las secciones"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        # Configuración válida no debe lanzar excepción
        config.validate()
        
        # Configuración inválida debe lanzar excepción
        config.database.backup_interval_hours = 0
        config.image_processing.max_image_size = 32
        config.matching.min_matches = 2
        
        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate()
        
        error_message = str(exc_info.value)
        assert "backup_interval_hours" in error_message
        assert "max_image_size" in error_message
        assert "min_matches" in error_message
    
    @patch('config.unified_config.Path.cwd')
    def test_update_config(self, mock_cwd):
        """Prueba actualización de configuración"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        # Actualizar sección válida
        config.update_config('database', sqlite_path='new/path.db', backup_enabled=False)
        
        assert config.database.sqlite_path == 'new/path.db'
        assert config.database.backup_enabled is False
        
        # Actualizar con valores inválidos debe lanzar excepción
        with pytest.raises(ConfigValidationError):
            config.update_config('database', backup_interval_hours=0)
        
        # Sección inexistente debe lanzar excepción
        with pytest.raises(ValueError):
            config.update_config('nonexistent', some_value=123)
    
    @patch('config.unified_config.Path.cwd')
    def test_reset_to_defaults(self, mock_cwd):
        """Prueba reseteo a valores por defecto"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        # Modificar valores
        config.database.sqlite_path = "custom/path.db"
        config.gui.theme = "dark"
        
        # Resetear sección específica
        config.reset_to_defaults('database')
        assert config.database.sqlite_path == "database/ballistics.db"
        assert config.gui.theme == "dark"  # No debe cambiar
        
        # Resetear toda la configuración
        config.reset_to_defaults()
        assert config.gui.theme == "default"
    
    @patch('config.unified_config.Path.cwd')
    def test_export_config(self, mock_cwd):
        """Prueba exportación de configuración"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        # Exportar como YAML
        yaml_file = config.export_config("yaml", str(self.temp_path / "test_config.yaml"))
        assert Path(yaml_file).exists()
        
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        assert 'database' in yaml_data
        assert 'gui' in yaml_data
        assert 'matching' in yaml_data
        
        # Exportar como JSON
        json_file = config.export_config("json", str(self.temp_path / "test_config.json"))
        assert Path(json_file).exists()
        
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        assert 'database' in json_data
        assert 'gui' in json_data
        assert 'matching' in json_data

    @patch('config.unified_config.Path.cwd')
    def test_default_config_file_mapping(self, mock_cwd):
        """Prueba mapeo de archivos de configuración por entorno"""
        mock_cwd.return_value = self.temp_path

        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()

        # Sin variable de entorno -> development
        with patch.dict(os.environ, {}, clear=True):
            cfg_dev = UnifiedConfig()
            assert cfg_dev.environment == Environment.DEVELOPMENT
            assert cfg_dev.config_file == "config/unified_config.yaml"

        # testing
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'testing'}, clear=True):
            cfg_test = UnifiedConfig()
            assert cfg_test.environment == Environment.TESTING
            assert cfg_test.config_file == "config/unified_config_testing.yaml"

        # production
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'production'}, clear=True):
            cfg_prod = UnifiedConfig()
            assert cfg_prod.environment == Environment.PRODUCTION
            assert cfg_prod.config_file == "config/unified_config_production.yaml"

        # Sinónimos
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'prod'}, clear=True):
            cfg_prod_syn = UnifiedConfig()
            assert cfg_prod_syn.environment == Environment.PRODUCTION
            assert cfg_prod_syn.config_file == "config/unified_config_production.yaml"

        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'test'}, clear=True):
            cfg_test_syn = UnifiedConfig()
            assert cfg_test_syn.environment == Environment.TESTING
            assert cfg_test_syn.config_file == "config/unified_config_testing.yaml"

        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'dev'}, clear=True):
            cfg_dev_syn = UnifiedConfig()
            assert cfg_dev_syn.environment == Environment.DEVELOPMENT
            assert cfg_dev_syn.config_file == "config/unified_config.yaml"

    @patch('config.unified_config.Path.cwd')
    def test_environment_yaml_loading_enum_conversion(self, mock_cwd):
        """Prueba carga de YAML por entorno y conversión de enums"""
        mock_cwd.return_value = self.temp_path

        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()

        # Crear YAML específico de testing con valores distintivos
        config_dir = self.temp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        testing_yaml = config_dir / "unified_config_testing.yaml"
        yaml_data = {
            'database': {
                'sqlite_path': 'database/testing.db',
                'faiss_index_path': 'database/faiss_index'
            },
            'gui': {
                'theme': 'dark',
                'language': 'en'
            },
            'matching': {
                'algorithm': 'SIFT',  # string que debe convertirse a enum
                'level': 'advanced',   # string que debe convertirse a enum
                'min_matches': 12
            }
        }
        with open(testing_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f)

        # Establecer entorno testing y cargar desde YAML temporal explícito
        with patch.dict(os.environ, {'SIGeC-Balistica_ENV': 'testing'}, clear=True):
            cfg = UnifiedConfig(config_file=str(testing_yaml))

        # Verificar que se leyó el archivo de testing
        assert cfg.environment == Environment.TESTING
        assert Path(cfg.config_file) == testing_yaml

        # Verificar conversión de enums y valores aplicados
        assert cfg.matching.algorithm == AlgorithmType.SIFT
        assert cfg.matching.level == MatchingLevel.ADVANCED
        assert cfg.matching.min_matches == 12
        assert cfg.gui.theme == 'dark'
        assert cfg.gui.language == 'en'

        # Guardar y verificar que se serializa como strings
        cfg.save_config()
        with open(testing_yaml, 'r', encoding='utf-8') as f:
            saved = yaml.safe_load(f)
        assert isinstance(saved['matching']['algorithm'], str)
        assert isinstance(saved['matching']['level'], str)
        assert saved['matching']['algorithm'] == 'SIFT'
        assert saved['matching']['level'] == 'advanced'
    
    @patch('config.unified_config.Path.cwd')
    def test_get_absolute_path(self, mock_cwd):
        """Prueba conversión de rutas relativas a absolutas"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        config = UnifiedConfig()
        
        # Ruta relativa
        relative_path = "database/ballistics.db"
        absolute_path = config.get_absolute_path(relative_path)
        expected_path = self.temp_path / relative_path
        assert absolute_path == expected_path
        
        # Ruta ya absoluta
        abs_path = "/absolute/path/file.db"
        result_path = config.get_absolute_path(abs_path)
        assert result_path == Path(abs_path)

class TestGlobalFunctions:
    """Pruebas para funciones globales"""
    
    def setup_method(self):
        """Configuración para cada prueba"""
        reset_unified_config()
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        reset_unified_config()
    
    @patch('config.unified_config.UnifiedConfig')
    def test_get_unified_config_singleton(self, mock_unified_config):
        """Prueba que get_unified_config funciona como singleton"""
        mock_instance = MagicMock()
        mock_unified_config.return_value = mock_instance
        
        # Primera llamada debe crear instancia
        config1 = get_unified_config()
        assert config1 == mock_instance
        assert mock_unified_config.call_count == 1
        
        # Segunda llamada debe retornar la misma instancia
        config2 = get_unified_config()
        assert config2 == mock_instance
        assert mock_unified_config.call_count == 1  # No debe crear nueva instancia
        
        # Forzar recarga debe crear nueva instancia
        config3 = get_unified_config(force_reload=True)
        assert config3 == mock_instance
        assert mock_unified_config.call_count == 2
    
    @patch('config.unified_config.get_unified_config')
    def test_convenience_functions(self, mock_get_config):
        """Prueba funciones de conveniencia"""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config
        
        # Probar todas las funciones de conveniencia
        get_database_config()
        mock_config.database
        
        get_image_processing_config()
        mock_config.image_processing
        
        get_matching_config()
        mock_config.matching
        
        get_gui_config()
        mock_config.gui
        
        get_logging_config()
        mock_config.logging
        
        get_deep_learning_config()
        mock_config.deep_learning
        
        get_nist_config()
        mock_config.nist

class TestLegacyMigration:
    """Pruebas para migración de configuraciones legacy"""
    
    def setup_method(self):
        """Configuración para cada prueba"""
        reset_unified_config()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Limpiar cualquier configuración existente en el directorio temporal
        config_dir = self.temp_path / "config"
        if config_dir.exists():
            import shutil
            shutil.rmtree(config_dir)
        
        # También limpiar el archivo de configuración global si existe
        global_config = Path("get_project_root()/config/unified_config.yaml")
        if global_config.exists():
            global_config.unlink()
    
    def teardown_method(self):
        """Limpieza después de cada prueba"""
        reset_unified_config()
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
    
    @patch('config.unified_config.Path.cwd')
    def test_legacy_config_migration(self, mock_cwd):
        """Prueba migración de config.yaml legacy"""
        mock_cwd.return_value = self.temp_path
        
        # Crear archivos indicadores
        (self.temp_path / "main.py").touch()
        (self.temp_path / "gui").mkdir()
        (self.temp_path / "matching").mkdir()
        
        # Crear config.yaml legacy
        legacy_config = {
            'database': {
                'sqlite_path': 'legacy/path.db',
                'backup_enabled': False
            },
            'gui': {
                'theme': 'dark',
                'language': 'en'
            }
        }
        
        legacy_file = self.temp_path / "config.yaml"
        with open(legacy_file, 'w') as f:
            yaml.dump(legacy_config, f)
        
        # Crear configuración unificada (debe migrar automáticamente)
        config = UnifiedConfig(auto_migrate=True)
        
        # Verificar que los valores fueron migrados
        assert config.database.sqlite_path == 'legacy/path.db'
        assert config.database.backup_enabled is False
        assert config.gui.theme == 'dark'
        assert config.gui.language == 'en'
        
        # Verificar que se creó respaldo
        backup_files = list(config.backup_dir.glob("config.yaml_*"))
        assert len(backup_files) > 0

if __name__ == "__main__":
    # Ejecutar pruebas
    pytest.main([__file__, "-v", "--tb=short"])