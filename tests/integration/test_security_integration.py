#!/usr/bin/env python3
"""
Tests de Integración de Seguridad para el Sistema SIGeC-Balistica
================================================================

Este módulo contiene tests específicos para validar aspectos de seguridad
del sistema, incluyendo autenticación, autorización, validación de entrada,
y protección contra vulnerabilidades comunes.

Autor: SIGeC-Balistica Team
Fecha: 2024
"""

import pytest
import tempfile
import shutil
import os
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import base64
import secrets

# Imports del sistema
try:
    from core.unified_pipeline import UnifiedPipeline
    from database.database_manager import DatabaseManager
    from config.unified_config import UnifiedConfig
    from utils.security_manager import SecurityManager
    from api.authentication import AuthenticationManager
    from api.authorization import AuthorizationManager
    from common.test_helpers import TestImageGenerator
    from utils.input_validator import InputValidator
    from utils.file_handler import SecureFileHandler
except ImportError as e:
    pytest.skip(f"Dependencias de seguridad no disponibles: {e}", allow_module_level=True)

class TestSecurityIntegration:
    """Tests de integración de seguridad"""
    
    @pytest.fixture(autouse=True)
    def setup_security_test_environment(self):
        """Configura entorno seguro para tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.secure_test_dir = self.temp_dir / "security_tests"
        self.secure_test_dir.mkdir(exist_ok=True, mode=0o700)  # Solo propietario
        
        # Configuración de seguridad
        self.security_config = self._create_security_config()
        
        # Crear imágenes de prueba
        self.image_generator = TestImageGenerator()
        self._create_test_images()
        
        # Configurar managers de seguridad
        self.security_manager = SecurityManager(self.security_config)
        self.auth_manager = AuthenticationManager(self.security_config)
        self.authz_manager = AuthorizationManager(self.security_config)
        
        yield
        
        # Limpieza segura
        self._secure_cleanup()
    
    def _create_security_config(self) -> Dict[str, Any]:
        """Crea configuración de seguridad"""
        return {
            'security': {
                'enable_authentication': True,
                'enable_authorization': True,
                'enable_input_validation': True,
                'enable_file_validation': True,
                'enable_audit_logging': True,
                'session_timeout': 3600,  # 1 hora
                'max_login_attempts': 3,
                'password_policy': {
                    'min_length': 8,
                    'require_uppercase': True,
                    'require_lowercase': True,
                    'require_numbers': True,
                    'require_special': True
                },
                'encryption': {
                    'algorithm': 'AES-256-GCM',
                    'key_derivation': 'PBKDF2',
                    'iterations': 100000
                },
                'file_validation': {
                    'max_file_size': 50 * 1024 * 1024,  # 50MB
                    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.tiff'],
                    'scan_for_malware': True,
                    'validate_headers': True
                }
            },
            'database': {
                'connection_string': f'sqlite:///{self.temp_dir}/secure_test.db',
                'enable_encryption': True,
                'enable_audit_trail': True,
                'connection_timeout': 30
            },
            'api': {
                'enable_rate_limiting': True,
                'rate_limit': '100/hour',
                'enable_cors': False,
                'allowed_origins': [],
                'enable_https_only': True
            },
            'logging': {
                'enable_security_logging': True,
                'log_level': 'INFO',
                'log_file': str(self.temp_dir / 'security.log'),
                'log_sensitive_data': False
            }
        }
    
    def _create_test_images(self):
        """Crea imágenes de prueba seguras"""
        self.test_images = {
            'valid': [],
            'malicious': [],
            'oversized': []
        }
        
        # Imágenes válidas
        for i in range(3):
            img_path = self.secure_test_dir / f"valid_{i}.jpg"
            image = self.image_generator.create_ballistic_image(
                width=512, height=384,
                features=['striations'],
                noise_level=0.1
            )
            self.image_generator.save_image(image, img_path)
            self.test_images['valid'].append(str(img_path))
        
        # Archivo "malicioso" (simulado)
        malicious_path = self.secure_test_dir / "malicious.jpg"
        with open(malicious_path, 'wb') as f:
            # Escribir header JPEG válido pero contenido sospechoso
            f.write(b'\xFF\xD8\xFF\xE0')  # JPEG header
            f.write(b'<script>alert("xss")</script>' * 100)  # Contenido sospechoso
        self.test_images['malicious'].append(str(malicious_path))
        
        # Archivo muy grande
        oversized_path = self.secure_test_dir / "oversized.jpg"
        with open(oversized_path, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0')  # JPEG header
            f.write(b'0' * (60 * 1024 * 1024))  # 60MB
        self.test_images['oversized'].append(str(oversized_path))
    
    def _secure_cleanup(self):
        """Limpieza segura de archivos temporales"""
        try:
            # Sobrescribir archivos sensibles antes de eliminar
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.exists():
                        # Sobrescribir con datos aleatorios
                        size = file_path.stat().st_size
                        with open(file_path, 'wb') as f:
                            f.write(secrets.token_bytes(size))
            
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignorar errores en limpieza
    
    def test_authentication_integration(self):
        """Test de integración de autenticación"""
        # Crear usuario de prueba
        test_user = {
            'username': 'test_user',
            'password': 'SecurePass123!',
            'role': 'analyst',
            'permissions': ['read', 'analyze']
        }
        
        # Registrar usuario
        user_id = self.auth_manager.register_user(
            username=test_user['username'],
            password=test_user['password'],
            role=test_user['role']
        )
        
        assert user_id is not None, "Registro de usuario falló"
        
        # Test de login exitoso
        session_token = self.auth_manager.authenticate(
            username=test_user['username'],
            password=test_user['password']
        )
        
        assert session_token is not None, "Autenticación falló"
        assert len(session_token) >= 32, "Token de sesión muy corto"
        
        # Validar sesión
        user_info = self.auth_manager.validate_session(session_token)
        assert user_info is not None, "Validación de sesión falló"
        assert user_info['username'] == test_user['username'], "Usuario incorrecto en sesión"
        
        # Test de login con credenciales incorrectas
        invalid_token = self.auth_manager.authenticate(
            username=test_user['username'],
            password='wrong_password'
        )
        
        assert invalid_token is None, "Autenticación debería fallar con password incorrecto"
        
        # Test de bloqueo por intentos fallidos
        for _ in range(4):  # Exceder max_login_attempts
            self.auth_manager.authenticate(
                username=test_user['username'],
                password='wrong_password'
            )
        
        # Ahora incluso con password correcto debería fallar
        blocked_token = self.auth_manager.authenticate(
            username=test_user['username'],
            password=test_user['password']
        )
        
        assert blocked_token is None, "Usuario debería estar bloqueado"
        
        print("✅ Integración de autenticación validada")
    
    def test_authorization_integration(self):
        """Test de integración de autorización"""
        # Crear usuarios con diferentes roles
        analyst_user = self.auth_manager.register_user(
            username='analyst', password='AnalystPass123!', role='analyst'
        )
        admin_user = self.auth_manager.register_user(
            username='admin', password='AdminPass123!', role='admin'
        )
        
        # Autenticar usuarios
        analyst_token = self.auth_manager.authenticate('analyst', 'AnalystPass123!')
        admin_token = self.auth_manager.authenticate('admin', 'AdminPass123!')
        
        # Test de autorización para análisis (ambos deberían poder)
        analyst_can_analyze = self.authz_manager.check_permission(
            session_token=analyst_token,
            resource='analysis',
            action='execute'
        )
        
        admin_can_analyze = self.authz_manager.check_permission(
            session_token=admin_token,
            resource='analysis',
            action='execute'
        )
        
        assert analyst_can_analyze, "Analista debería poder ejecutar análisis"
        assert admin_can_analyze, "Admin debería poder ejecutar análisis"
        
        # Test de autorización para administración (solo admin)
        analyst_can_admin = self.authz_manager.check_permission(
            session_token=analyst_token,
            resource='system',
            action='configure'
        )
        
        admin_can_admin = self.authz_manager.check_permission(
            session_token=admin_token,
            resource='system',
            action='configure'
        )
        
        assert not analyst_can_admin, "Analista NO debería poder configurar sistema"
        assert admin_can_admin, "Admin debería poder configurar sistema"
        
        # Test con token inválido
        invalid_permission = self.authz_manager.check_permission(
            session_token='invalid_token',
            resource='analysis',
            action='execute'
        )
        
        assert not invalid_permission, "Token inválido no debería tener permisos"
        
        print("✅ Integración de autorización validada")
    
    def test_input_validation_integration(self):
        """Test de integración de validación de entrada"""
        validator = InputValidator(self.security_config)
        
        # Test de validación de archivos
        valid_image = self.test_images['valid'][0]
        malicious_file = self.test_images['malicious'][0]
        oversized_file = self.test_images['oversized'][0]
        
        # Archivo válido debería pasar
        is_valid, message = validator.validate_image_file(valid_image)
        assert is_valid, f"Archivo válido rechazado: {message}"
        
        # Archivo malicioso debería ser rechazado
        is_malicious, message = validator.validate_image_file(malicious_file)
        assert not is_malicious, f"Archivo malicioso aceptado: {message}"
        
        # Archivo muy grande debería ser rechazado
        is_oversized, message = validator.validate_image_file(oversized_file)
        assert not is_oversized, f"Archivo muy grande aceptado: {message}"
        
        # Test de validación de parámetros
        valid_params = {
            'threshold': 0.8,
            'algorithm': 'sift',
            'max_features': 1000
        }
        
        invalid_params = {
            'threshold': 1.5,  # Fuera de rango
            'algorithm': '<script>alert("xss")</script>',  # Malicioso
            'max_features': -100  # Valor inválido
        }
        
        # Parámetros válidos
        params_valid, message = validator.validate_analysis_parameters(valid_params)
        assert params_valid, f"Parámetros válidos rechazados: {message}"
        
        # Parámetros inválidos
        params_invalid, message = validator.validate_analysis_parameters(invalid_params)
        assert not params_invalid, f"Parámetros inválidos aceptados: {message}"
        
        print("✅ Integración de validación de entrada validada")
    
    def test_secure_file_handling_integration(self):
        """Test de integración de manejo seguro de archivos"""
        file_handler = SecureFileHandler(self.security_config)
        
        valid_image = self.test_images['valid'][0]
        
        # Test de carga segura
        image_data, metadata = file_handler.load_image_securely(valid_image)
        
        assert image_data is not None, "Carga segura de imagen falló"
        assert 'file_hash' in metadata, "Hash de archivo no generado"
        assert 'file_size' in metadata, "Tamaño de archivo no registrado"
        assert 'validation_status' in metadata, "Estado de validación no registrado"
        
        # Test de almacenamiento seguro
        secure_path = file_handler.store_image_securely(
            image_data=image_data,
            original_filename=Path(valid_image).name,
            user_id='test_user'
        )
        
        assert secure_path is not None, "Almacenamiento seguro falló"
        assert Path(secure_path).exists(), "Archivo seguro no creado"
        
        # Verificar que el archivo tiene permisos restrictivos
        file_stat = Path(secure_path).stat()
        file_mode = oct(file_stat.st_mode)[-3:]
        assert file_mode == '600', f"Permisos de archivo inseguros: {file_mode}"
        
        # Test de eliminación segura
        file_handler.delete_file_securely(secure_path)
        assert not Path(secure_path).exists(), "Archivo no eliminado"
        
        print("✅ Integración de manejo seguro de archivos validada")
    
    def test_encrypted_database_integration(self):
        """Test de integración con base de datos encriptada"""
        config = self.security_config.copy()
        config['database']['enable_encryption'] = True
        
        db_manager = DatabaseManager(config['database'])
        
        # Datos sensibles de prueba
        sensitive_data = {
            'case_id': 'CASE_001',
            'evidence_hash': hashlib.sha256(b'evidence_data').hexdigest(),
            'analysis_result': {
                'match_probability': 0.95,
                'features_matched': 150,
                'algorithm_used': 'deep_learning'
            },
            'investigator': 'Detective Smith',
            'classification': 'CONFIDENTIAL'
        }
        
        # Almacenar datos encriptados
        stored_id = db_manager.store_encrypted_case(sensitive_data)
        assert stored_id is not None, "Almacenamiento encriptado falló"
        
        # Recuperar y verificar datos
        retrieved_data = db_manager.retrieve_encrypted_case(stored_id)
        assert retrieved_data is not None, "Recuperación de datos encriptados falló"
        assert retrieved_data['case_id'] == sensitive_data['case_id'], "Datos corruptos tras encriptación"
        
        # Verificar que los datos están encriptados en BD
        raw_data = db_manager.get_raw_case_data(stored_id)
        assert raw_data != sensitive_data, "Datos no encriptados en BD"
        
        # Test de búsqueda en datos encriptados
        search_results = db_manager.search_encrypted_cases(
            field='case_id',
            value='CASE_001'
        )
        
        assert len(search_results) > 0, "Búsqueda en datos encriptados falló"
        assert search_results[0]['case_id'] == 'CASE_001', "Resultado de búsqueda incorrecto"
        
        print("✅ Integración con base de datos encriptada validada")
    
    def test_audit_logging_integration(self):
        """Test de integración de logging de auditoría"""
        # Configurar pipeline con auditoría
        config = self.security_config.copy()
        config['logging']['enable_security_logging'] = True
        
        pipeline = UnifiedPipeline(config)
        
        # Simular sesión de usuario
        user_token = self.auth_manager.authenticate('analyst', 'AnalystPass123!')
        
        # Ejecutar análisis con auditoría
        with patch.object(pipeline, '_get_current_user_token', return_value=user_token):
            result = pipeline.process_images(
                self.test_images['valid'][0],
                self.test_images['valid'][1]
            )
        
        assert result.success, "Procesamiento con auditoría falló"
        
        # Verificar logs de auditoría
        log_file = Path(self.security_config['logging']['log_file'])
        assert log_file.exists(), "Archivo de log no creado"
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Verificar eventos de auditoría
        assert 'AUDIT' in log_content, "Eventos de auditoría no registrados"
        assert 'analyst' in log_content, "Usuario no registrado en auditoría"
        assert 'process_images' in log_content, "Acción no registrada en auditoría"
        
        # Verificar que datos sensibles no están en logs
        assert 'SecurePass123!' not in log_content, "Password en logs de auditoría"
        
        print("✅ Integración de logging de auditoría validada")
    
    def test_session_security_integration(self):
        """Test de integración de seguridad de sesiones"""
        # Crear sesión
        user_token = self.auth_manager.authenticate('analyst', 'AnalystPass123!')
        assert user_token is not None, "Autenticación inicial falló"
        
        # Verificar propiedades de seguridad del token
        assert len(user_token) >= 32, "Token muy corto"
        assert user_token.isalnum() or '-' in user_token or '_' in user_token, "Token con caracteres inseguros"
        
        # Test de timeout de sesión
        with patch('time.time', return_value=time.time() + 3700):  # +1 hora y 2 minutos
            expired_user = self.auth_manager.validate_session(user_token)
            assert expired_user is None, "Sesión expirada no invalidada"
        
        # Test de invalidación manual
        self.auth_manager.invalidate_session(user_token)
        invalidated_user = self.auth_manager.validate_session(user_token)
        assert invalidated_user is None, "Sesión no invalidada manualmente"
        
        # Test de sesiones concurrentes
        token1 = self.auth_manager.authenticate('analyst', 'AnalystPass123!')
        token2 = self.auth_manager.authenticate('analyst', 'AnalystPass123!')
        
        # Ambas sesiones deberían ser válidas inicialmente
        user1 = self.auth_manager.validate_session(token1)
        user2 = self.auth_manager.validate_session(token2)
        
        assert user1 is not None, "Primera sesión concurrente inválida"
        assert user2 is not None, "Segunda sesión concurrente inválida"
        
        # Logout de una sesión no debería afectar la otra
        self.auth_manager.logout(token1)
        
        user1_after = self.auth_manager.validate_session(token1)
        user2_after = self.auth_manager.validate_session(token2)
        
        assert user1_after is None, "Primera sesión no cerrada"
        assert user2_after is not None, "Segunda sesión afectada incorrectamente"
        
        print("✅ Integración de seguridad de sesiones validada")
    
    def test_rate_limiting_integration(self):
        """Test de integración de limitación de tasa"""
        if not hasattr(self, 'api_client'):
            pytest.skip("Cliente API no disponible")
        
        # Configurar límite bajo para testing
        config = self.security_config.copy()
        config['api']['rate_limit'] = '5/minute'
        
        api_client = self.api_client(config)
        
        # Realizar requests dentro del límite
        successful_requests = 0
        for i in range(5):
            response = api_client.post('/api/analyze', {
                'image1': self.test_images['valid'][0],
                'image2': self.test_images['valid'][1]
            })
            
            if response.status_code == 200:
                successful_requests += 1
        
        assert successful_requests == 5, f"Solo {successful_requests}/5 requests exitosos"
        
        # El siguiente request debería ser limitado
        limited_response = api_client.post('/api/analyze', {
            'image1': self.test_images['valid'][0],
            'image2': self.test_images['valid'][1]
        })
        
        assert limited_response.status_code == 429, "Rate limiting no aplicado"
        assert 'rate limit' in limited_response.text.lower(), "Mensaje de rate limit no presente"
        
        print("✅ Integración de limitación de tasa validada")
    
    def test_security_headers_integration(self):
        """Test de integración de headers de seguridad"""
        if not hasattr(self, 'api_client'):
            pytest.skip("Cliente API no disponible")
        
        config = self.security_config.copy()
        api_client = self.api_client(config)
        
        response = api_client.get('/api/health')
        
        # Verificar headers de seguridad
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'"
        }
        
        for header, expected_value in security_headers.items():
            assert header in response.headers, f"Header de seguridad {header} faltante"
            if expected_value:
                assert expected_value in response.headers[header], f"Valor incorrecto para {header}"
        
        # Verificar que información sensible no se expone
        sensitive_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']
        for header in sensitive_headers:
            assert header not in response.headers, f"Header sensible {header} expuesto"
        
        print("✅ Integración de headers de seguridad validada")
    
    def test_data_sanitization_integration(self):
        """Test de integración de sanitización de datos"""
        config = self.security_config.copy()
        pipeline = UnifiedPipeline(config)
        
        # Datos con contenido potencialmente malicioso
        malicious_metadata = {
            'case_name': '<script>alert("xss")</script>',
            'investigator': 'John"; DROP TABLE cases; --',
            'notes': '../../etc/passwd',
            'tags': ['normal', '<img src=x onerror=alert(1)>', 'evidence']
        }
        
        # Procesar con datos maliciosos
        result = pipeline.process_images_with_metadata(
            image1=self.test_images['valid'][0],
            image2=self.test_images['valid'][1],
            metadata=malicious_metadata
        )
        
        assert result.success, "Procesamiento con sanitización falló"
        
        # Verificar que los datos fueron sanitizados
        sanitized_metadata = result.metadata
        
        assert '<script>' not in sanitized_metadata['case_name'], "XSS no sanitizado"
        assert 'DROP TABLE' not in sanitized_metadata['investigator'], "SQL injection no sanitizado"
        assert '../' not in sanitized_metadata['notes'], "Path traversal no sanitizado"
        
        # Verificar sanitización en arrays
        for tag in sanitized_metadata['tags']:
            assert '<' not in tag and '>' not in tag, f"Tag malicioso no sanitizado: {tag}"
        
        print("✅ Integración de sanitización de datos validada")

if __name__ == "__main__":
    # Ejecutar tests si se llama directamente
    pytest.main([__file__, "-v", "--tb=short"])