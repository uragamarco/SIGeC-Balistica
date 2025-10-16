---
title: Manual de Instalación
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: operadores, devops
toc: true
tags:
  - instalación
  - requisitos
  - producción
  - configuración
---

# SIGeC-Balisticar - Manual de Instalación

## Tabla de Contenidos
1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación en Desarrollo](#instalación-en-desarrollo)
3. [Instalación en Producción](#instalación-en-producción)
4. [Configuración](#configuración)
5. [Verificación de la Instalación](#verificación-de-la-instalación)
6. [Solución de Problemas](#solución-de-problemas)

---

## Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Linux (Ubuntu 20.04+, CentOS 8+, Debian 10+)
- **Python**: 3.8 o superior
- **Memoria RAM**: 4 GB mínimo, 8 GB recomendado
- **Espacio en Disco**: 10 GB mínimo, 50 GB recomendado
- **CPU**: 2 núcleos mínimo, 4 núcleos recomendado

### Requisitos Recomendados para Producción
- **Sistema Operativo**: Ubuntu 22.04 LTS o CentOS Stream 9
- **Python**: 3.10 o superior
- **Memoria RAM**: 16 GB o más
- **Espacio en Disco**: 100 GB o más (SSD recomendado)
- **CPU**: 8 núcleos o más
- **Base de Datos**: PostgreSQL 13+ (servidor dedicado recomendado)

### Dependencias del Sistema
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y libpq-dev build-essential
sudo apt install -y git curl wget
sudo apt install -y nginx supervisor

# CentOS/RHEL
sudo dnf update
sudo dnf install -y python3 python3-pip python3-devel
sudo dnf install -y postgresql postgresql-server postgresql-contrib
sudo dnf install -y postgresql-devel gcc gcc-c++
sudo dnf install -y git curl wget
sudo dnf install -y nginx supervisor
```

---

## Instalación en Desarrollo

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-organizacion/SIGeC-Balisticar.git
cd SIGeC-Balisticar
```

### 2. Crear Entorno Virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar Base de Datos (Desarrollo)
```bash
# Crear base de datos PostgreSQL
sudo -u postgres createdb sigec_balistica_dev
sudo -u postgres createuser sigec_dev_user
sudo -u postgres psql -c "ALTER USER sigec_dev_user PASSWORD 'dev_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sigec_balistica_dev TO sigec_dev_user;"
```

### 5. Configurar Variables de Entorno
```bash
# Crear archivo .env
cat > .env << EOF
# Desarrollo
ENVIRONMENT=development
DEBUG=true

# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sigec_balistica_dev
DB_USER=sigec_dev_user
DB_PASSWORD=dev_password

# Aplicación
APP_HOST=127.0.0.1
APP_PORT=5000

# Seguridad (generar claves reales)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
EOF
```

### 6. Inicializar Base de Datos
```bash
python scripts/init_database.py
```

### 7. Ejecutar Aplicación
```bash
python main.py
```

La aplicación estará disponible en: http://127.0.0.1:5000

---

## Instalación en Producción

### 1. Preparar el Sistema

#### Crear Usuario del Sistema
```bash
sudo useradd -r -s /bin/false sigec
sudo mkdir -p /opt/sigec-balistica
sudo mkdir -p /var/lib/sigec-balistica
sudo mkdir -p /var/log/sigec-balistica
sudo mkdir -p /var/backups/sigec-balistica
sudo chown -R sigec:sigec /opt/sigec-balistica /var/lib/sigec-balistica /var/log/sigec-balistica /var/backups/sigec-balistica
```

#### Configurar PostgreSQL
```bash
# Inicializar PostgreSQL (CentOS/RHEL)
sudo postgresql-setup --initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Crear base de datos y usuario
sudo -u postgres createdb sigec_balistica_prod
sudo -u postgres createuser sigec_prod_user
sudo -u postgres psql -c "ALTER USER sigec_prod_user PASSWORD 'SECURE_PASSWORD_HERE';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sigec_balistica_prod TO sigec_prod_user;"

# Configurar autenticación
sudo nano /var/lib/pgsql/data/pg_hba.conf
# Agregar línea:
# local   sigec_balistica_prod    sigec_prod_user                     md5

sudo systemctl restart postgresql
```

### 2. Instalar la Aplicación

#### Clonar y Configurar
```bash
cd /opt/sigec-balistica
sudo -u sigec git clone https://github.com/tu-organizacion/SIGeC-Balisticar.git .
sudo -u sigec python3 -m venv venv
sudo -u sigec /opt/sigec-balistica/venv/bin/pip install --upgrade pip
sudo -u sigec /opt/sigec-balistica/venv/bin/pip install -r requirements.txt
```

#### Configurar Variables de Entorno
```bash
sudo -u sigec tee /opt/sigec-balistica/.env << EOF
# Producción
ENVIRONMENT=production
DEBUG=false

# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sigec_balistica_prod
DB_USER=sigec_prod_user
DB_PASSWORD=SECURE_PASSWORD_HERE

# Aplicación
APP_HOST=0.0.0.0
APP_PORT=8000

# Seguridad (generar claves seguras)
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 32)

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/sigec-balistica/app.log
EOF
```

### 3. Configurar Servicios del Sistema

#### Crear Servicio Systemd
```bash
sudo tee /etc/systemd/system/sigec-balistica.service << EOF
[Unit]
Description=SIGeC-Balistica Application
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=sigec
Group=sigec
WorkingDirectory=/opt/sigec-balistica
Environment=PATH=/opt/sigec-balistica/venv/bin
EnvironmentFile=/opt/sigec-balistica/.env
ExecStart=/opt/sigec-balistica/venv/bin/python main.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10

# Límites de recursos
LimitNOFILE=65536
LimitNPROC=4096

# Seguridad
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/sigec-balistica /var/log/sigec-balistica /tmp

[Install]
WantedBy=multi-user.target
EOF
```

#### Habilitar y Iniciar Servicio
```bash
sudo systemctl daemon-reload
sudo systemctl enable sigec-balistica
sudo systemctl start sigec-balistica
sudo systemctl status sigec-balistica
```

### 4. Configurar Nginx (Proxy Reverso)

```bash
sudo tee /etc/nginx/sites-available/sigec-balistica << EOF
server {
    listen 80;
    server_name tu-dominio.com;
    
    # Redirigir HTTP a HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name tu-dominio.com;
    
    # Certificados SSL (configurar con Let's Encrypt o certificados propios)
    ssl_certificate /etc/ssl/certs/sigec-balistica.crt;
    ssl_certificate_key /etc/ssl/private/sigec-balistica.key;
    
    # Configuración SSL
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Headers de seguridad
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # Configuración del proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffers
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # Archivos estáticos
    location /static/ {
        alias /opt/sigec-balistica/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Límites de subida
    client_max_body_size 100M;
    
    # Logs
    access_log /var/log/nginx/sigec-balistica.access.log;
    error_log /var/log/nginx/sigec-balistica.error.log;
}
EOF

# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/sigec-balistica /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 5. Configurar Certificados SSL con Let's Encrypt

```bash
# Instalar Certbot
sudo apt install certbot python3-certbot-nginx  # Ubuntu/Debian
sudo dnf install certbot python3-certbot-nginx  # CentOS/RHEL

# Obtener certificado
sudo certbot --nginx -d tu-dominio.com

# Configurar renovación automática
sudo crontab -e
# Agregar línea:
# 0 12 * * * /usr/bin/certbot renew --quiet
```

### 6. Configurar Monitoreo y Logs

#### Configurar Logrotate
```bash
sudo tee /etc/logrotate.d/sigec-balistica << EOF
/var/log/sigec-balistica/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 sigec sigec
    postrotate
        systemctl reload sigec-balistica
    endscript
}
EOF
```

#### Configurar Supervisor (Alternativo a Systemd)
```bash
sudo tee /etc/supervisor/conf.d/sigec-balistica.conf << EOF
[program:sigec-balistica]
command=/opt/sigec-balistica/venv/bin/python main.py
directory=/opt/sigec-balistica
user=sigec
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/sigec-balistica/supervisor.log
environment=PATH="/opt/sigec-balistica/venv/bin"
EOF

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sigec-balistica
```

---

## Configuración

### Archivo de Configuración Principal

Crear `/etc/sigec-balistica/production.json`:

```json
{
  "environment": "production",
  "debug": false,
  "app_host": "0.0.0.0",
  "app_port": 8000,
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "sigec_balistica_prod",
    "username": "sigec_prod_user",
    "ssl_mode": "require",
    "connection_pool_size": 20,
    "max_overflow": 30
  },
  "security": {
    "jwt_expiration_hours": 24,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 30,
    "session_timeout_minutes": 60,
    "https_only": true
  },
  "performance": {
    "cache_enabled": true,
    "cache_max_size": 2000,
    "cache_ttl_seconds": 3600,
    "max_workers": 8,
    "batch_size": 10,
    "max_image_size_mb": 50
  },
  "logging": {
    "level": "INFO",
    "file_path": "/var/log/sigec-balistica/app.log",
    "max_file_size_mb": 100,
    "backup_count": 5,
    "enable_console": false
  },
  "monitoring": {
    "enabled": true,
    "metrics_port": 9090,
    "health_check_port": 8080,
    "collect_system_metrics": true
  },
  "backup": {
    "enabled": true,
    "backup_directory": "/var/backups/sigec-balistica",
    "db_backup_schedule": "0 2 * * *",
    "db_backup_retention_days": 30
  }
}
```

### Variables de Entorno Importantes

```bash
# Aplicación
ENVIRONMENT=production
DEBUG=false
APP_HOST=0.0.0.0
APP_PORT=8000

# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sigec_balistica_prod
DB_USER=sigec_prod_user
DB_PASSWORD=secure_password_here

# Seguridad
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/sigec-balistica/app.log

# Monitoreo
MONITORING_ENABLED=true
METRICS_PORT=9090
```

---

## Configuración

### Sistema de Configuración en Capas

SIGeC-Balisticar v2.0 implementa un **Sistema de Configuración en Capas** que permite herencia y sobrescritura de valores entre diferentes entornos.

#### Estructura de Archivos de Configuración

```
config/
├── config_layers.yaml          # Configuración principal en capas
├── config_manager.py           # Gestor unificado de configuración
├── layered_config_manager.py   # Gestor de capas
└── unified_config.yaml         # Configuración legacy (migración automática)
```

#### Archivo Principal: config/config_layers.yaml

```yaml
# Configuración base (común a todos los entornos)
base:
  project:
    name: "SIGeC-Balisticar"
    version: "2.0.0"
  
  database:
    type: "unified"
    host: "localhost"
    port: 5432
    name: "ballistic_db"
    ssl_enabled: false
  
  gui:
    theme: "modern"
    enable_gpu: true
    window:
      width: 1200
      height: 800
  
  image_processing:
    roi_detection: "watershed"
    feature_extraction: "orb_sift_hybrid"
    max_image_size_mb: 50
  
  matching:
    algorithm: "unified_matcher"
    cmc_threshold: 8
  
  logging:
    level: "INFO"
    enable_console: true

# Configuración de testing (hereda de base + overrides)
testing:
  database:
    name: "ballistic_test_db"
  
  logging:
    level: "DEBUG"
    enable_console: true

# Configuración de producción (hereda de base + overrides)
production:
  database:
    host: "prod-db-server"
    ssl_enabled: true
    connection_pool_size: 20
  
  gui:
    enable_gpu: false  # Deshabilitado en servidor
  
  logging:
    level: "INFO"
    file: "/var/log/sigec/app.log"
    enable_console: false
  
  performance:
    cache_enabled: true
    max_workers: 8
    batch_size: 10
```

#### Variables de Entorno

El sistema soporta sobrescritura mediante variables de entorno con el prefijo `SIGEC_`:

```bash
# Configuración de entorno
export SIGEC_ENVIRONMENT="production"

# Sobrescritura de base de datos
export SIGEC_DATABASE_HOST="prod-db-server.company.com"
export SIGEC_DATABASE_PORT="5432"
export SIGEC_DATABASE_NAME="sigec_balistica_prod"
export SIGEC_DATABASE_SSL_ENABLED="true"

# Sobrescritura de GUI
export SIGEC_GUI_THEME="dark"
export SIGEC_GUI_ENABLE_GPU="false"

# Sobrescritura de logging
export SIGEC_LOGGING_LEVEL="INFO"
export SIGEC_LOGGING_FILE="/var/log/sigec/app.log"
```

#### Configuración para Desarrollo

```bash
# Crear archivo de configuración local
cp config/config_layers.yaml config/config_layers_local.yaml

# Editar configuración local
nano config/config_layers_local.yaml

# Usar configuración local
export SIGEC_CONFIG_FILE="config/config_layers_local.yaml"
export SIGEC_ENVIRONMENT="base"
```

#### Configuración para Testing

```bash
# Variables para testing
export SIGEC_ENVIRONMENT="testing"
export SIGEC_DATABASE_NAME="ballistic_test_db"
export SIGEC_LOGGING_LEVEL="DEBUG"

# Ejecutar tests
python -m pytest tests/ -v
```

#### Configuración para Producción

```bash
# Variables de producción
export SIGEC_ENVIRONMENT="production"
export SIGEC_DATABASE_HOST="your-prod-db-host"
export SIGEC_DATABASE_USER="sigec_prod_user"
export SIGEC_DATABASE_PASSWORD="secure_password_here"
export SIGEC_LOGGING_FILE="/var/log/sigec-balistica/app.log"

# Archivo de entorno para systemd
sudo tee /opt/sigec-balistica/.env << EOF
SIGEC_ENVIRONMENT=production
SIGEC_DATABASE_HOST=localhost
SIGEC_DATABASE_PORT=5432
SIGEC_DATABASE_NAME=sigec_balistica_prod
SIGEC_DATABASE_USER=sigec_prod_user
SIGEC_DATABASE_PASSWORD=secure_password_here
SIGEC_LOGGING_LEVEL=INFO
SIGEC_LOGGING_FILE=/var/log/sigec-balistica/app.log
EOF
```

#### Migración desde Configuraciones Anteriores

El sistema incluye **migración automática** desde configuraciones legacy:

```python
# El sistema detecta automáticamente archivos legacy:
# - config.yaml
# - gui_config.yaml  
# - test_config.yaml
# - production_config.yaml

# Y los migra al nuevo formato automáticamente
from config.config_manager import get_unified_manager

config_manager = get_unified_manager()
config = config_manager.load_config("unified", "production")
```

#### Validación de Configuración

```bash
# Validar configuración actual
python -c "
from config.config_manager import get_unified_manager
config_manager = get_unified_manager()

# Validar diferentes entornos
for env in ['base', 'testing', 'production']:
    try:
        config = config_manager.load_config('unified', env)
        print(f'✓ Configuración {env} válida')
    except Exception as e:
        print(f'✗ Error en configuración {env}: {e}')
"
```

### Configuración Legacy (Producción)

Para compatibilidad con versiones anteriores, también se mantiene soporte para configuración JSON:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "sigec_balistica_prod",
    "user": "sigec_prod_user",
    "password": "secure_password_here",
    "ssl_mode": "require",
    "connection_pool_size": 20,
    "connection_timeout": 30
  },
  "security": {
    "secret_key": "your-secret-key-here",
    "jwt_secret_key": "your-jwt-secret-here",
    "session_timeout": 3600,
    "max_login_attempts": 5,
    "lockout_duration": 900
  },
  "performance": {

### 1. Ejecutar Validador de Despliegue

```bash
cd /opt/sigec-balistica
sudo -u sigec /opt/sigec-balistica/venv/bin/python production/deployment_validator.py
```

### 2. Verificar Servicios

```bash
# Estado del servicio principal
sudo systemctl status sigec-balistica

# Estado de la base de datos
sudo systemctl status postgresql

# Estado de Nginx
sudo systemctl status nginx

# Verificar puertos
sudo netstat -tlnp | grep -E ':(80|443|8000|5432)'
```

### 3. Pruebas de Conectividad

```bash
# Verificar aplicación
curl -k https://tu-dominio.com/health

# Verificar métricas (si está habilitado)
curl http://localhost:9090/metrics

# Verificar logs
sudo tail -f /var/log/sigec-balistica/app.log
```

### 4. Pruebas Funcionales

```bash
# Ejecutar tests de integración
cd /opt/sigec-balistica
sudo -u sigec /opt/sigec-balistica/venv/bin/python -m pytest tests/integration/ -v

# Verificar GUI (si está disponible)
sudo -u sigec /opt/sigec-balistica/venv/bin/python verify_gui_services.py
```

---

## Solución de Problemas

### Problemas Comunes

#### 1. Error de Conexión a Base de Datos
```bash
# Verificar estado de PostgreSQL
sudo systemctl status postgresql

# Verificar configuración de conexión
sudo -u postgres psql -c "\l" | grep sigec

# Verificar logs de PostgreSQL
sudo tail -f /var/log/postgresql/postgresql-*.log
```

#### 2. Problemas de Permisos
```bash
# Verificar permisos de directorios
ls -la /opt/sigec-balistica
ls -la /var/lib/sigec-balistica
ls -la /var/log/sigec-balistica

# Corregir permisos
sudo chown -R sigec:sigec /opt/sigec-balistica
sudo chown -R sigec:sigec /var/lib/sigec-balistica
sudo chown -R sigec:sigec /var/log/sigec-balistica
```

#### 3. Problemas de Dependencias
```bash
# Verificar entorno virtual
sudo -u sigec /opt/sigec-balistica/venv/bin/python -c "import sys; print(sys.path)"

# Reinstalar dependencias
sudo -u sigec /opt/sigec-balistica/venv/bin/pip install -r requirements.txt --force-reinstall
```

#### 4. Problemas de SSL/HTTPS
```bash
# Verificar certificados
sudo certbot certificates

# Renovar certificados
sudo certbot renew --dry-run

# Verificar configuración de Nginx
sudo nginx -t
```

### Logs Importantes

```bash
# Logs de la aplicación
sudo tail -f /var/log/sigec-balistica/app.log

# Logs del sistema
sudo journalctl -u sigec-balistica -f

# Logs de Nginx
sudo tail -f /var/log/nginx/sigec-balistica.error.log
sudo tail -f /var/log/nginx/sigec-balistica.access.log

# Logs de PostgreSQL
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### Comandos de Diagnóstico

```bash
# Verificar recursos del sistema
htop
df -h
free -h

# Verificar conectividad de red
netstat -tlnp
ss -tlnp

# Verificar procesos
ps aux | grep sigec
ps aux | grep nginx
ps aux | grep postgres
```

### Contacto y Soporte

Para soporte técnico:
- Email: soporte@sigec-balistica.com
- Documentación: https://docs.sigec-balistica.com
- Issues: https://github.com/tu-organizacion/SIGeC-Balisticar/issues

---

## Actualizaciones

### Proceso de Actualización

```bash
# 1. Hacer backup
sudo -u sigec /opt/sigec-balistica/scripts/backup.sh

# 2. Detener servicio
sudo systemctl stop sigec-balistica

# 3. Actualizar código
cd /opt/sigec-balistica
sudo -u sigec git pull origin main

# 4. Actualizar dependencias
sudo -u sigec /opt/sigec-balistica/venv/bin/pip install -r requirements.txt --upgrade

# 5. Ejecutar migraciones (si las hay)
sudo -u sigec /opt/sigec-balistica/venv/bin/python scripts/migrate.py

# 6. Iniciar servicio
sudo systemctl start sigec-balistica

# 7. Verificar funcionamiento
sudo systemctl status sigec-balistica
```

---

*Última actualización: $(date)*
*Versión del documento: 1.0*