#!/usr/bin/env python3
"""
Sistema de notificaciones inteligente para SEACABAr.
Maneja notificaciones de errores, alertas y eventos del sistema.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

class NotificationType(Enum):
    """Tipos de notificaciones."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"

class NotificationChannel(Enum):
    """Canales de notificación."""
    LOG = "log"
    GUI = "gui"
    EMAIL = "email"
    WEBHOOK = "webhook"
    FILE = "file"

@dataclass
class Notification:
    """Representa una notificación del sistema."""
    id: str
    timestamp: datetime
    type: NotificationType
    title: str
    message: str
    component: str
    channels: List[NotificationChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    expires_at: Optional[datetime] = None

class NotificationManager:
    """Gestor principal del sistema de notificaciones."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.notifications: List[Notification] = []
        self.subscribers: Dict[NotificationType, List[Callable]] = {}
        self.channels: Dict[NotificationChannel, Callable] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Configuración
        self.max_notifications = 1000
        self.cleanup_interval = 3600  # 1 hora
        self.last_cleanup = time.time()
        
        # Configurar canales por defecto
        self._setup_default_channels()
        
        # Cargar configuración si existe
        if config_path:
            self._load_config(config_path)
    
    def _setup_default_channels(self):
        """Configura los canales de notificación por defecto."""
        self.channels[NotificationChannel.LOG] = self._send_log_notification
        self.channels[NotificationChannel.GUI] = self._send_gui_notification
        self.channels[NotificationChannel.FILE] = self._send_file_notification
    
    def send_notification(self, 
                         notification_type: NotificationType,
                         title: str,
                         message: str,
                         component: str,
                         channels: Optional[List[NotificationChannel]] = None,
                         metadata: Optional[Dict[str, Any]] = None,
                         expires_in: Optional[timedelta] = None) -> str:
        """Envía una notificación."""
        
        notification_id = f"{component}_{int(time.time() * 1000)}"
        
        # Determinar canales por defecto según el tipo
        if channels is None:
            channels = self._get_default_channels(notification_type)
        
        # Calcular expiración
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
        
        notification = Notification(
            id=notification_id,
            timestamp=datetime.now(),
            type=notification_type,
            title=title,
            message=message,
            component=component,
            channels=channels,
            metadata=metadata or {},
            expires_at=expires_at
        )
        
        # Almacenar notificación
        with self.lock:
            self.notifications.append(notification)
            self._cleanup_old_notifications()
        
        # Enviar por canales configurados
        self._dispatch_notification(notification)
        
        # Notificar suscriptores
        self._notify_subscribers(notification)
        
        return notification_id
    
    def _get_default_channels(self, notification_type: NotificationType) -> List[NotificationChannel]:
        """Obtiene los canales por defecto según el tipo de notificación."""
        
        if notification_type == NotificationType.CRITICAL:
            return [NotificationChannel.LOG, NotificationChannel.GUI, NotificationChannel.FILE]
        elif notification_type == NotificationType.ERROR:
            return [NotificationChannel.LOG, NotificationChannel.GUI]
        elif notification_type == NotificationType.WARNING:
            return [NotificationChannel.LOG]
        else:
            return [NotificationChannel.LOG]
    
    def _dispatch_notification(self, notification: Notification):
        """Despacha la notificación a los canales configurados."""
        
        for channel in notification.channels:
            if channel in self.channels:
                try:
                    self.channels[channel](notification)
                except Exception as e:
                    self.logger.error(f"Error enviando notificación por {channel.value}: {e}")
    
    def _notify_subscribers(self, notification: Notification):
        """Notifica a los suscriptores del tipo de notificación."""
        
        subscribers = self.subscribers.get(notification.type, [])
        for subscriber in subscribers:
            try:
                subscriber(notification)
            except Exception as e:
                self.logger.error(f"Error notificando suscriptor: {e}")
    
    def _send_log_notification(self, notification: Notification):
        """Envía notificación por log."""
        
        level_map = {
            NotificationType.INFO: logging.INFO,
            NotificationType.WARNING: logging.WARNING,
            NotificationType.ERROR: logging.ERROR,
            NotificationType.CRITICAL: logging.CRITICAL,
            NotificationType.SUCCESS: logging.INFO
        }
        
        level = level_map.get(notification.type, logging.INFO)
        self.logger.log(level, f"[{notification.component}] {notification.title}: {notification.message}")
    
    def _send_gui_notification(self, notification: Notification):
        """Envía notificación a la GUI."""
        # Aquí se implementaría la integración con la GUI
        # Por ahora, solo registramos que se envió
        self.logger.debug(f"Notificación GUI enviada: {notification.title}")
    
    def _send_file_notification(self, notification: Notification):
        """Guarda notificación en archivo."""
        
        notifications_file = Path("logs/notifications.json")
        notifications_file.parent.mkdir(exist_ok=True)
        
        notification_data = {
            'id': notification.id,
            'timestamp': notification.timestamp.isoformat(),
            'type': notification.type.value,
            'title': notification.title,
            'message': notification.message,
            'component': notification.component,
            'metadata': notification.metadata
        }
        
        try:
            # Leer notificaciones existentes
            existing_notifications = []
            if notifications_file.exists():
                with open(notifications_file, 'r', encoding='utf-8') as f:
                    existing_notifications = json.load(f)
            
            # Agregar nueva notificación
            existing_notifications.append(notification_data)
            
            # Mantener solo las últimas 100 notificaciones
            if len(existing_notifications) > 100:
                existing_notifications = existing_notifications[-100:]
            
            # Guardar
            with open(notifications_file, 'w', encoding='utf-8') as f:
                json.dump(existing_notifications, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error guardando notificación en archivo: {e}")
    
    def subscribe(self, notification_type: NotificationType, callback: Callable):
        """Suscribe un callback a un tipo de notificación."""
        
        with self.lock:
            if notification_type not in self.subscribers:
                self.subscribers[notification_type] = []
            self.subscribers[notification_type].append(callback)
    
    def unsubscribe(self, notification_type: NotificationType, callback: Callable):
        """Desuscribe un callback de un tipo de notificación."""
        
        with self.lock:
            if notification_type in self.subscribers:
                try:
                    self.subscribers[notification_type].remove(callback)
                except ValueError:
                    pass
    
    def acknowledge_notification(self, notification_id: str) -> bool:
        """Marca una notificación como reconocida."""
        
        with self.lock:
            for notification in self.notifications:
                if notification.id == notification_id:
                    notification.acknowledged = True
                    return True
        return False
    
    def get_notifications(self, 
                         component: Optional[str] = None,
                         notification_type: Optional[NotificationType] = None,
                         acknowledged: Optional[bool] = None,
                         limit: int = 50) -> List[Notification]:
        """Obtiene notificaciones filtradas."""
        
        with self.lock:
            filtered = self.notifications.copy()
        
        # Aplicar filtros
        if component:
            filtered = [n for n in filtered if n.component == component]
        
        if notification_type:
            filtered = [n for n in filtered if n.type == notification_type]
        
        if acknowledged is not None:
            filtered = [n for n in filtered if n.acknowledged == acknowledged]
        
        # Ordenar por timestamp descendente y limitar
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered[:limit]
    
    def _cleanup_old_notifications(self):
        """Limpia notificaciones antiguas y expiradas."""
        
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        now = datetime.now()
        
        # Remover notificaciones expiradas
        self.notifications = [
            n for n in self.notifications 
            if n.expires_at is None or n.expires_at > now
        ]
        
        # Mantener solo las más recientes si hay demasiadas
        if len(self.notifications) > self.max_notifications:
            self.notifications.sort(key=lambda x: x.timestamp, reverse=True)
            self.notifications = self.notifications[:self.max_notifications]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de notificaciones."""
        
        with self.lock:
            total = len(self.notifications)
            acknowledged = sum(1 for n in self.notifications if n.acknowledged)
            
            by_type = {}
            by_component = {}
            
            for notification in self.notifications:
                # Por tipo
                type_key = notification.type.value
                by_type[type_key] = by_type.get(type_key, 0) + 1
                
                # Por componente
                comp_key = notification.component
                by_component[comp_key] = by_component.get(comp_key, 0) + 1
        
        return {
            'total_notifications': total,
            'acknowledged': acknowledged,
            'unacknowledged': total - acknowledged,
            'by_type': by_type,
            'by_component': by_component,
            'last_24h': len([
                n for n in self.notifications 
                if n.timestamp > datetime.now() - timedelta(days=1)
            ])
        }

# Instancia global del gestor de notificaciones
_notification_manager = None

def get_notification_manager() -> NotificationManager:
    """Obtiene la instancia global del gestor de notificaciones."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager

# Funciones de conveniencia
def notify_info(title: str, message: str, component: str, **kwargs) -> str:
    """Envía una notificación de información."""
    return get_notification_manager().send_notification(
        NotificationType.INFO, title, message, component, **kwargs
    )

def notify_warning(title: str, message: str, component: str, **kwargs) -> str:
    """Envía una notificación de advertencia."""
    return get_notification_manager().send_notification(
        NotificationType.WARNING, title, message, component, **kwargs
    )

def notify_error(title: str, message: str, component: str, **kwargs) -> str:
    """Envía una notificación de error."""
    return get_notification_manager().send_notification(
        NotificationType.ERROR, title, message, component, **kwargs
    )

def notify_critical(title: str, message: str, component: str, **kwargs) -> str:
    """Envía una notificación crítica."""
    return get_notification_manager().send_notification(
        NotificationType.CRITICAL, title, message, component, **kwargs
    )

def notify_success(title: str, message: str, component: str, **kwargs) -> str:
    """Envía una notificación de éxito."""
    return get_notification_manager().send_notification(
        NotificationType.SUCCESS, title, message, component, **kwargs
    )