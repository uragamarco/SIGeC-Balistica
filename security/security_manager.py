#!/usr/bin/env python3
"""
Sistema de Seguridad Avanzado para SEACABAr.
Proporciona autenticación, autorización, auditoría y protección de datos.
"""

import hashlib
import secrets
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from functools import wraps
import hmac
import base64

# Configurar logging
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de seguridad del sistema."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class AccessLevel(Enum):
    """Niveles de acceso de usuarios."""
    GUEST = "guest"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class AuditEventType(Enum):
    """Tipos de eventos de auditoría."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_EVENT = "system_event"

@dataclass
class SecurityContext:
    """Contexto de seguridad para operaciones."""
    user_id: str
    session_id: str
    access_level: AccessLevel
    permissions: Set[str]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = None
    expires_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=8)

@dataclass
class AuditEvent:
    """Evento de auditoría."""
    event_id: str
    event_type: AuditEventType
    user_id: str
    session_id: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, denied
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = None
    risk_score: float = 0.0

@dataclass
class SecurityPolicy:
    """Política de seguridad."""
    name: str
    description: str
    rules: Dict[str, Any]
    security_level: SecurityLevel
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

class SecurityManager:
    """Gestor principal de seguridad."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Inicializar gestor de seguridad."""
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'security'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de seguridad
        self.session_timeout = self.config.get('session_timeout', 28800)  # 8 horas
        self.max_login_attempts = self.config.get('max_login_attempts', 5)
        self.lockout_duration = self.config.get('lockout_duration', 900)  # 15 minutos
        self.password_min_length = self.config.get('password_min_length', 12)
        self.require_mfa = self.config.get('require_mfa', False)
        
        # Estado interno
        self._sessions: Dict[str, SecurityContext] = {}
        self._users: Dict[str, Dict[str, Any]] = {}
        self._policies: Dict[str, SecurityPolicy] = {}
        self._audit_events: List[AuditEvent] = []
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._locked_accounts: Dict[str, datetime] = {}
        self._permissions: Dict[AccessLevel, Set[str]] = {}
        
        # Threading
        self._lock = threading.RLock()
        
        # Inicializar sistema
        self._initialize_system()
    
    def _initialize_system(self):
        """Inicializar sistema de seguridad."""
        
        # Cargar datos existentes
        self._load_users()
        self._load_policies()
        self._load_audit_log()
        
        # Configurar permisos por defecto
        self._setup_default_permissions()
        
        # Configurar políticas por defecto
        self._setup_default_policies()
        
        logger.info("Sistema de seguridad inicializado")
    
    def _setup_default_permissions(self):
        """Configurar permisos por defecto para cada nivel de acceso."""
        
        self._permissions = {
            AccessLevel.GUEST: {
                'view_public_data',
                'basic_operations'
            },
            AccessLevel.USER: {
                'view_public_data',
                'view_internal_data',
                'basic_operations',
                'user_operations',
                'create_content',
                'modify_own_content'
            },
            AccessLevel.OPERATOR: {
                'view_public_data',
                'view_internal_data',
                'view_confidential_data',
                'basic_operations',
                'user_operations',
                'operator_operations',
                'create_content',
                'modify_content',
                'system_monitoring'
            },
            AccessLevel.ADMIN: {
                'view_public_data',
                'view_internal_data',
                'view_confidential_data',
                'view_restricted_data',
                'basic_operations',
                'user_operations',
                'operator_operations',
                'admin_operations',
                'create_content',
                'modify_content',
                'delete_content',
                'system_monitoring',
                'system_configuration',
                'user_management'
            },
            AccessLevel.SUPER_ADMIN: {
                'view_public_data',
                'view_internal_data',
                'view_confidential_data',
                'view_restricted_data',
                'view_top_secret_data',
                'basic_operations',
                'user_operations',
                'operator_operations',
                'admin_operations',
                'super_admin_operations',
                'create_content',
                'modify_content',
                'delete_content',
                'system_monitoring',
                'system_configuration',
                'user_management',
                'security_management',
                'audit_access'
            }
        }
    
    def _setup_default_policies(self):
        """Configurar políticas de seguridad por defecto."""
        
        default_policies = [
            SecurityPolicy(
                name="password_policy",
                description="Política de contraseñas seguras",
                security_level=SecurityLevel.INTERNAL,
                rules={
                    'min_length': self.password_min_length,
                    'require_uppercase': True,
                    'require_lowercase': True,
                    'require_numbers': True,
                    'require_special_chars': True,
                    'max_age_days': 90,
                    'history_count': 5
                }
            ),
            SecurityPolicy(
                name="session_policy",
                description="Política de gestión de sesiones",
                security_level=SecurityLevel.INTERNAL,
                rules={
                    'timeout_seconds': self.session_timeout,
                    'max_concurrent_sessions': 3,
                    'require_secure_transport': True,
                    'idle_timeout_seconds': 1800  # 30 minutos
                }
            ),
            SecurityPolicy(
                name="access_control_policy",
                description="Política de control de acceso",
                security_level=SecurityLevel.CONFIDENTIAL,
                rules={
                    'max_login_attempts': self.max_login_attempts,
                    'lockout_duration_seconds': self.lockout_duration,
                    'require_mfa': self.require_mfa,
                    'ip_whitelist_enabled': False,
                    'allowed_ip_ranges': []
                }
            ),
            SecurityPolicy(
                name="audit_policy",
                description="Política de auditoría y logging",
                security_level=SecurityLevel.RESTRICTED,
                rules={
                    'log_all_access': True,
                    'log_data_access': True,
                    'log_failed_attempts': True,
                    'retention_days': 365,
                    'real_time_alerts': True
                }
            )
        ]
        
        for policy in default_policies:
            self._policies[policy.name] = policy
    
    def create_user(self, user_id: str, password: str, access_level: AccessLevel,
                   email: Optional[str] = None, full_name: Optional[str] = None) -> bool:
        """Crear nuevo usuario."""
        
        with self._lock:
            if user_id in self._users:
                logger.warning(f"Usuario {user_id} ya existe")
                return False
            
            # Validar contraseña
            if not self._validate_password(password):
                logger.warning(f"Contraseña no cumple políticas para usuario {user_id}")
                return False
            
            # Hash de contraseña
            password_hash = self._hash_password(password)
            
            # Crear usuario
            user_data = {
                'user_id': user_id,
                'password_hash': password_hash,
                'access_level': access_level.value,
                'email': email,
                'full_name': full_name,
                'created_at': datetime.now().isoformat(),
                'last_login': None,
                'failed_attempts': 0,
                'locked_until': None,
                'mfa_enabled': False,
                'mfa_secret': None,
                'active': True
            }
            
            self._users[user_id] = user_data
            self._save_users()
            
            # Auditar creación
            self._audit_event(
                AuditEventType.SYSTEM_EVENT,
                user_id="system",
                session_id=None,
                resource="user_management",
                action="create_user",
                result="success",
                details={'created_user': user_id, 'access_level': access_level.value}
            )
            
            logger.info(f"Usuario {user_id} creado exitosamente")
            return True
    
    def authenticate_user(self, user_id: str, password: str,
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None) -> Optional[SecurityContext]:
        """Autenticar usuario y crear contexto de seguridad."""
        
        with self._lock:
            # Verificar si la cuenta está bloqueada
            if self._is_account_locked(user_id):
                self._audit_event(
                    AuditEventType.LOGIN_FAILURE,
                    user_id=user_id,
                    session_id=None,
                    resource="authentication",
                    action="login",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={'reason': 'account_locked'}
                )
                return None
            
            # Verificar usuario existe
            if user_id not in self._users:
                self._record_failed_attempt(user_id)
                self._audit_event(
                    AuditEventType.LOGIN_FAILURE,
                    user_id=user_id,
                    session_id=None,
                    resource="authentication",
                    action="login",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={'reason': 'user_not_found'}
                )
                return None
            
            user_data = self._users[user_id]
            
            # Verificar usuario activo
            if not user_data.get('active', True):
                self._audit_event(
                    AuditEventType.LOGIN_FAILURE,
                    user_id=user_id,
                    session_id=None,
                    resource="authentication",
                    action="login",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={'reason': 'account_disabled'}
                )
                return None
            
            # Verificar contraseña
            if not self._verify_password(password, user_data['password_hash']):
                self._record_failed_attempt(user_id)
                self._audit_event(
                    AuditEventType.LOGIN_FAILURE,
                    user_id=user_id,
                    session_id=None,
                    resource="authentication",
                    action="login",
                    result="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    details={'reason': 'invalid_password'}
                )
                return None
            
            # Limpiar intentos fallidos
            if user_id in self._failed_attempts:
                del self._failed_attempts[user_id]
            
            # Crear sesión
            session_id = self._generate_session_id()
            access_level = AccessLevel(user_data['access_level'])
            permissions = self._permissions.get(access_level, set())
            
            context = SecurityContext(
                user_id=user_id,
                session_id=session_id,
                access_level=access_level,
                permissions=permissions.copy(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self._sessions[session_id] = context
            
            # Actualizar último login
            user_data['last_login'] = datetime.now().isoformat()
            self._save_users()
            
            # Auditar login exitoso
            self._audit_event(
                AuditEventType.LOGIN_SUCCESS,
                user_id=user_id,
                session_id=session_id,
                resource="authentication",
                action="login",
                result="success",
                ip_address=ip_address,
                user_agent=user_agent,
                details={'access_level': access_level.value}
            )
            
            logger.info(f"Usuario {user_id} autenticado exitosamente")
            return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validar sesión existente."""
        
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            context = self._sessions[session_id]
            
            # Verificar expiración
            if datetime.now() > context.expires_at:
                self._invalidate_session(session_id)
                return None
            
            return context
    
    def check_permission(self, session_id: str, permission: str,
                        resource: Optional[str] = None) -> bool:
        """Verificar si una sesión tiene un permiso específico."""
        
        context = self.validate_session(session_id)
        if not context:
            return False
        
        has_permission = permission in context.permissions
        
        # Auditar acceso
        self._audit_event(
            AuditEventType.ACCESS_GRANTED if has_permission else AuditEventType.ACCESS_DENIED,
            user_id=context.user_id,
            session_id=session_id,
            resource=resource or "unknown",
            action="check_permission",
            result="success" if has_permission else "denied",
            details={'permission': permission}
        )
        
        return has_permission
    
    def logout_user(self, session_id: str) -> bool:
        """Cerrar sesión de usuario."""
        
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            context = self._sessions[session_id]
            
            # Auditar logout
            self._audit_event(
                AuditEventType.LOGOUT,
                user_id=context.user_id,
                session_id=session_id,
                resource="authentication",
                action="logout",
                result="success"
            )
            
            # Invalidar sesión
            self._invalidate_session(session_id)
            
            logger.info(f"Usuario {context.user_id} cerró sesión")
            return True
    
    def get_audit_events(self, user_id: Optional[str] = None,
                        event_type: Optional[AuditEventType] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: int = 100) -> List[AuditEvent]:
        """Obtener eventos de auditoría."""
        
        with self._lock:
            events = self._audit_events.copy()
            
            # Filtrar por usuario
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            
            # Filtrar por tipo
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Filtrar por fecha
            if start_date:
                events = [e for e in events if e.timestamp >= start_date]
            
            if end_date:
                events = [e for e in events if e.timestamp <= end_date]
            
            # Ordenar por timestamp descendente y limitar
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events[:limit]
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generar reporte de seguridad."""
        
        with self._lock:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Estadísticas de eventos
            recent_events = [e for e in self._audit_events if e.timestamp >= last_24h]
            weekly_events = [e for e in self._audit_events if e.timestamp >= last_7d]
            
            # Contar por tipo
            event_counts = {}
            for event in recent_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Intentos de login fallidos
            failed_logins = [e for e in recent_events 
                           if e.event_type == AuditEventType.LOGIN_FAILURE]
            
            # Sesiones activas
            active_sessions = len([s for s in self._sessions.values() 
                                 if s.expires_at > now])
            
            # Cuentas bloqueadas
            locked_accounts = len([u for u, until in self._locked_accounts.items() 
                                 if until > now])
            
            return {
                'timestamp': now.isoformat(),
                'summary': {
                    'total_users': len(self._users),
                    'active_sessions': active_sessions,
                    'locked_accounts': locked_accounts,
                    'total_policies': len(self._policies)
                },
                'recent_activity': {
                    'events_last_24h': len(recent_events),
                    'events_last_7d': len(weekly_events),
                    'failed_logins_24h': len(failed_logins),
                    'event_breakdown': event_counts
                },
                'security_status': {
                    'policies_enabled': len([p for p in self._policies.values() if p.enabled]),
                    'mfa_users': len([u for u in self._users.values() if u.get('mfa_enabled', False)]),
                    'password_policy_compliant': True  # Simplificado
                },
                'top_events': [
                    {
                        'event_type': e.event_type.value,
                        'user_id': e.user_id,
                        'timestamp': e.timestamp.isoformat(),
                        'resource': e.resource,
                        'result': e.result
                    }
                    for e in recent_events[:10]
                ]
            }
    
    # Métodos privados
    
    def _validate_password(self, password: str) -> bool:
        """Validar contraseña según políticas."""
        
        policy = self._policies.get('password_policy')
        if not policy or not policy.enabled:
            return len(password) >= self.password_min_length
        
        rules = policy.rules
        
        # Longitud mínima
        if len(password) < rules.get('min_length', 12):
            return False
        
        # Mayúsculas
        if rules.get('require_uppercase', True) and not any(c.isupper() for c in password):
            return False
        
        # Minúsculas
        if rules.get('require_lowercase', True) and not any(c.islower() for c in password):
            return False
        
        # Números
        if rules.get('require_numbers', True) and not any(c.isdigit() for c in password):
            return False
        
        # Caracteres especiales
        if rules.get('require_special_chars', True):
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash seguro de contraseña."""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verificar contraseña contra hash almacenado."""
        try:
            salt, hash_hex = stored_hash.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(password_hash.hex(), hash_hex)
        except Exception:
            return False
    
    def _generate_session_id(self) -> str:
        """Generar ID de sesión único."""
        return secrets.token_urlsafe(32)
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Verificar si cuenta está bloqueada."""
        if user_id in self._locked_accounts:
            unlock_time = self._locked_accounts[user_id]
            if datetime.now() < unlock_time:
                return True
            else:
                # Desbloquear cuenta
                del self._locked_accounts[user_id]
        
        return False
    
    def _record_failed_attempt(self, user_id: str):
        """Registrar intento fallido de login."""
        now = datetime.now()
        
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []
        
        # Limpiar intentos antiguos (últimos 15 minutos)
        cutoff = now - timedelta(minutes=15)
        self._failed_attempts[user_id] = [
            attempt for attempt in self._failed_attempts[user_id] 
            if attempt > cutoff
        ]
        
        # Agregar nuevo intento
        self._failed_attempts[user_id].append(now)
        
        # Verificar si debe bloquearse
        if len(self._failed_attempts[user_id]) >= self.max_login_attempts:
            self._locked_accounts[user_id] = now + timedelta(seconds=self.lockout_duration)
            logger.warning(f"Cuenta {user_id} bloqueada por múltiples intentos fallidos")
    
    def _invalidate_session(self, session_id: str):
        """Invalidar sesión."""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def _audit_event(self, event_type: AuditEventType, user_id: str,
                    session_id: Optional[str], resource: str, action: str,
                    result: str, ip_address: Optional[str] = None,
                    user_agent: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None):
        """Registrar evento de auditoría."""
        
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            result=result,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        self._audit_events.append(event)
        
        # Mantener solo los últimos 10000 eventos en memoria
        if len(self._audit_events) > 10000:
            self._audit_events = self._audit_events[-10000:]
        
        # Guardar en disco periódicamente
        if len(self._audit_events) % 100 == 0:
            self._save_audit_log()
    
    def _load_users(self):
        """Cargar usuarios desde disco."""
        users_file = self.data_dir / 'users.json'
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    self._users = json.load(f)
                logger.info(f"Cargados {len(self._users)} usuarios")
            except Exception as e:
                logger.error(f"Error cargando usuarios: {e}")
    
    def _save_users(self):
        """Guardar usuarios a disco."""
        users_file = self.data_dir / 'users.json'
        try:
            with open(users_file, 'w') as f:
                json.dump(self._users, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando usuarios: {e}")
    
    def _load_policies(self):
        """Cargar políticas desde disco."""
        policies_file = self.data_dir / 'policies.json'
        if policies_file.exists():
            try:
                with open(policies_file, 'r') as f:
                    policies_data = json.load(f)
                    for name, data in policies_data.items():
                        self._policies[name] = SecurityPolicy(
                            name=data['name'],
                            description=data['description'],
                            rules=data['rules'],
                            security_level=SecurityLevel(data['security_level']),
                            enabled=data.get('enabled', True),
                            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
                            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
                        )
                logger.info(f"Cargadas {len(self._policies)} políticas")
            except Exception as e:
                logger.error(f"Error cargando políticas: {e}")
    
    def _save_policies(self):
        """Guardar políticas a disco."""
        policies_file = self.data_dir / 'policies.json'
        try:
            policies_data = {}
            for name, policy in self._policies.items():
                policies_data[name] = {
                    'name': policy.name,
                    'description': policy.description,
                    'rules': policy.rules,
                    'security_level': policy.security_level.value,
                    'enabled': policy.enabled,
                    'created_at': policy.created_at.isoformat() if policy.created_at else None,
                    'updated_at': policy.updated_at.isoformat() if policy.updated_at else None
                }
            
            with open(policies_file, 'w') as f:
                json.dump(policies_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando políticas: {e}")
    
    def _load_audit_log(self):
        """Cargar log de auditoría desde disco."""
        audit_file = self.data_dir / 'audit.json'
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    audit_data = json.load(f)
                    for event_data in audit_data:
                        event = AuditEvent(
                            event_id=event_data['event_id'],
                            event_type=AuditEventType(event_data['event_type']),
                            user_id=event_data['user_id'],
                            session_id=event_data.get('session_id'),
                            resource=event_data['resource'],
                            action=event_data['action'],
                            result=event_data['result'],
                            timestamp=datetime.fromisoformat(event_data['timestamp']),
                            ip_address=event_data.get('ip_address'),
                            user_agent=event_data.get('user_agent'),
                            details=event_data.get('details', {}),
                            risk_score=event_data.get('risk_score', 0.0)
                        )
                        self._audit_events.append(event)
                logger.info(f"Cargados {len(self._audit_events)} eventos de auditoría")
            except Exception as e:
                logger.error(f"Error cargando auditoría: {e}")
    
    def _save_audit_log(self):
        """Guardar log de auditoría a disco."""
        audit_file = self.data_dir / 'audit.json'
        try:
            audit_data = []
            for event in self._audit_events:
                audit_data.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'resource': event.resource,
                    'action': event.action,
                    'result': event.result,
                    'timestamp': event.timestamp.isoformat(),
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent,
                    'details': event.details,
                    'risk_score': event.risk_score
                })
            
            with open(audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando auditoría: {e}")

# Decoradores de seguridad

def require_permission(permission: str, resource: str = None):
    """Decorador para requerir permiso específico."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Buscar session_id en argumentos
            session_id = kwargs.get('session_id')
            if not session_id and args:
                # Buscar en argumentos posicionales
                for arg in args:
                    if isinstance(arg, str) and len(arg) == 43:  # Longitud típica de session_id
                        session_id = arg
                        break
            
            if not session_id:
                raise PermissionError("Session ID requerido")
            
            # Obtener gestor de seguridad global
            security_manager = get_security_manager()
            if not security_manager.check_permission(session_id, permission, resource):
                raise PermissionError(f"Permiso '{permission}' requerido")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_access_level(min_level: AccessLevel):
    """Decorador para requerir nivel de acceso mínimo."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id')
            if not session_id and args:
                for arg in args:
                    if isinstance(arg, str) and len(arg) == 43:
                        session_id = arg
                        break
            
            if not session_id:
                raise PermissionError("Session ID requerido")
            
            security_manager = get_security_manager()
            context = security_manager.validate_session(session_id)
            
            if not context:
                raise PermissionError("Sesión inválida")
            
            # Verificar nivel de acceso
            access_levels = [AccessLevel.GUEST, AccessLevel.USER, AccessLevel.OPERATOR, 
                           AccessLevel.ADMIN, AccessLevel.SUPER_ADMIN]
            
            if access_levels.index(context.access_level) < access_levels.index(min_level):
                raise PermissionError(f"Nivel de acceso '{min_level.value}' requerido")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Instancia global
_security_manager: Optional[SecurityManager] = None

def get_security_manager() -> SecurityManager:
    """Obtener instancia global del gestor de seguridad."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def initialize_security(config: Dict[str, Any] = None) -> SecurityManager:
    """Inicializar sistema de seguridad."""
    global _security_manager
    _security_manager = SecurityManager(config)
    return _security_manager

# Funciones de conveniencia

def create_admin_user(user_id: str = "admin", password: str = None) -> bool:
    """Crear usuario administrador por defecto."""
    if password is None:
        password = secrets.token_urlsafe(16)
        print(f"Contraseña generada para admin: {password}")
    
    security_manager = get_security_manager()
    return security_manager.create_user(
        user_id=user_id,
        password=password,
        access_level=AccessLevel.SUPER_ADMIN,
        full_name="Administrador del Sistema"
    )

def authenticate(user_id: str, password: str, **kwargs) -> Optional[str]:
    """Autenticar usuario y retornar session_id."""
    security_manager = get_security_manager()
    context = security_manager.authenticate_user(user_id, password, **kwargs)
    return context.session_id if context else None

def check_access(session_id: str, permission: str, resource: str = None) -> bool:
    """Verificar acceso rápido."""
    security_manager = get_security_manager()
    return security_manager.check_permission(session_id, permission, resource)

if __name__ == "__main__":
    # Ejemplo de uso
    security_manager = initialize_security()
    
    # Crear usuario admin
    create_admin_user("admin", "SecurePassword123!")
    
    # Autenticar
    session_id = authenticate("admin", "SecurePassword123!")
    if session_id:
        print(f"Autenticado exitosamente. Session ID: {session_id}")
        
        # Verificar permisos
        if check_access(session_id, "system_configuration"):
            print("Acceso a configuración del sistema: PERMITIDO")
        
        # Generar reporte
        report = security_manager.get_security_report()
        print(f"Reporte de seguridad: {json.dumps(report, indent=2)}")
    else:
        print("Autenticación fallida")