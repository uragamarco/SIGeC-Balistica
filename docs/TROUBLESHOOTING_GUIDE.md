---
title: Guía de Solución de Problemas
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: soporte, devops
toc: true
tags:
  - troubleshooting
  - diagnóstico
  - rendimiento
  - base_de_datos
---

# SIGeC-Balisticar - Guía de Solución de Problemas

## Tabla de Contenidos
1. [Diagnóstico General](#diagnóstico-general)
2. [Problemas de Instalación](#problemas-de-instalación)
3. [Problemas de Base de Datos](#problemas-de-base-de-datos)
4. [Problemas de Procesamiento](#problemas-de-procesamiento)
5. [Problemas de Rendimiento](#problemas-de-rendimiento)
6. [Problemas de Interfaz](#problemas-de-interfaz)
7. [Problemas de Red](#problemas-de-red)
8. [Problemas de Seguridad](#problemas-de-seguridad)
9. [Herramientas de Diagnóstico](#herramientas-de-diagnóstico)
10. [Logs y Monitoreo](#logs-y-monitoreo)
11. [Contacto de Soporte](#contacto-de-soporte)

---

## Diagnóstico General

### Lista de Verificación Inicial

Antes de proceder con diagnósticos específicos, verifique:

```bash
# 1. Estado del sistema
systemctl status sigec-balistica

# 2. Recursos del sistema
free -h
df -h
top -n 1

# 3. Conectividad de red
ping -c 3 localhost
netstat -tlnp | grep :8000

# 4. Logs recientes
tail -n 50 /var/log/sigec-balistica/app.log
```

### Comandos de Diagnóstico Rápido

```bash
#!/bin/bash
# Script de diagnóstico rápido

echo "=== DIAGNÓSTICO SIGEC-BALISTICA ==="
echo "Fecha: $(date)"
echo

echo "1. Estado del servicio:"
systemctl is-active sigec-balistica
echo

echo "2. Uso de recursos:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memoria: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "Disco: $(df -h / | awk 'NR==2{printf "%s", $5}')"
echo

echo "3. Conectividad:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Aplicación responde"
else
    echo "❌ Aplicación no responde"
fi

echo "4. Base de datos:"
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "✅ PostgreSQL disponible"
else
    echo "❌ PostgreSQL no disponible"
fi

echo "5. Últimos errores:"
tail -n 5 /var/log/sigec-balistica/error.log
```

---

## Problemas de Instalación

### Error: Dependencias Faltantes

**Síntomas**:
```
ModuleNotFoundError: No module named 'cv2'
ImportError: No module named 'sklearn'
```

**Solución**:
```bash
# Instalar dependencias del sistema
sudo apt update
sudo apt install python3-dev python3-pip libopencv-dev

# Instalar dependencias de Python
pip3 install -r requirements.txt

# Verificar instalación
python3 -c "import cv2, sklearn, numpy; print('Dependencias OK')"
```

### Error: Permisos Insuficientes

**Síntomas**:
```
PermissionError: [Errno 13] Permission denied: '/var/log/sigec-balistica'
```

**Solución**:
```bash
# Crear directorios con permisos correctos
sudo mkdir -p /var/log/sigec-balistica
sudo mkdir -p /var/lib/sigec-balistica
sudo mkdir -p /etc/sigec-balistica

# Asignar propietario
sudo chown -R sigec:sigec /var/log/sigec-balistica
sudo chown -R sigec:sigec /var/lib/sigec-balistica
sudo chown -R sigec:sigec /etc/sigec-balistica

# Establecer permisos
sudo chmod 755 /var/log/sigec-balistica
sudo chmod 755 /var/lib/sigec-balistica
sudo chmod 750 /etc/sigec-balistica
```

### Error: Puerto en Uso

**Síntomas**:
```
OSError: [Errno 98] Address already in use
```

**Solución**:
```bash
# Identificar proceso usando el puerto
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# Terminar proceso si es necesario
sudo kill -9 <PID>

# O cambiar puerto en configuración
nano /etc/sigec-balistica/config.json
# Cambiar "port": 8000 por "port": 8001
```

### Error: Base de Datos No Inicializada

**Síntomas**:
```
psycopg2.OperationalError: FATAL: database "sigec_balistica" does not exist
```

**Solución**:
```bash
# Crear base de datos
sudo -u postgres createdb sigec_balistica
sudo -u postgres createuser sigec_user

# Asignar permisos
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sigec_balistica TO sigec_user;"

# Ejecutar migraciones
cd /opt/sigec-balistica
python3 manage.py migrate
```

---

## Problemas de Base de Datos

### Conexión Fallida

**Síntomas**:
```
psycopg2.OperationalError: could not connect to server
FATAL: password authentication failed for user "sigec_user"
```

**Diagnóstico**:
```bash
# Verificar estado de PostgreSQL
systemctl status postgresql

# Probar conexión manual
psql -h localhost -U sigec_user -d sigec_balistica

# Verificar configuración
sudo cat /etc/postgresql/*/main/pg_hba.conf | grep sigec
```

**Solución**:
```bash
# Reiniciar PostgreSQL
sudo systemctl restart postgresql

# Verificar/cambiar contraseña
sudo -u postgres psql
ALTER USER sigec_user PASSWORD 'nueva_contraseña';

# Actualizar configuración de la aplicación
nano /etc/sigec-balistica/config.json
```

### Rendimiento Lento de Consultas

**Síntomas**:
- Consultas tardan más de 5 segundos
- Timeouts frecuentes
- Alto uso de CPU en PostgreSQL

**Diagnóstico**:
```sql
-- Consultas lentas
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Índices faltantes
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;
```

**Solución**:
```sql
-- Crear índices necesarios
CREATE INDEX idx_samples_type ON samples(type);
CREATE INDEX idx_samples_created_at ON samples(created_at);
CREATE INDEX idx_features_sample_id ON features(sample_id);

-- Actualizar estadísticas
ANALYZE;

-- Configurar PostgreSQL
-- En postgresql.conf:
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

### Corrupción de Datos

**Síntomas**:
```
ERROR: invalid page in block 1234 of relation base/16384/12345
```

**Diagnóstico**:
```bash
# Verificar integridad
sudo -u postgres pg_dump sigec_balistica > /dev/null

# Verificar filesystem
sudo fsck /dev/sda1
```

**Solución**:
```bash
# Detener aplicación
sudo systemctl stop sigec-balistica

# Reparar base de datos
sudo -u postgres reindexdb sigec_balistica

# Restaurar desde backup si es necesario
sudo -u postgres pg_restore -d sigec_balistica /backup/sigec_backup.sql

# Reiniciar servicios
sudo systemctl start postgresql
sudo systemctl start sigec-balistica
```

---

## Problemas de Procesamiento

### Error al Procesar Imágenes

**Síntomas**:
```
cv2.error: OpenCV(4.5.4) error: (-215:Assertion failed) !empty() in function 'cv::imread'
```

**Diagnóstico**:
```python
import cv2
import os

def diagnose_image(image_path):
    print(f"Archivo existe: {os.path.exists(image_path)}")
    print(f"Tamaño: {os.path.getsize(image_path)} bytes")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ No se puede leer la imagen")
    else:
        print(f"✅ Imagen: {img.shape}")
```

**Solución**:
```bash
# Verificar formatos soportados
python3 -c "import cv2; print(cv2.getBuildInformation())"

# Convertir imagen si es necesario
convert imagen.webp imagen.jpg

# Verificar permisos
chmod 644 imagen.jpg

# Verificar integridad
file imagen.jpg
identify imagen.jpg
```

### Algoritmos No Disponibles

**Síntomas**:
```
AttributeError: module 'cv2' has no attribute 'SIFT_create'
```

**Solución**:
```bash
# Instalar OpenCV con contrib
pip3 uninstall opencv-python
pip3 install opencv-contrib-python

# Verificar algoritmos disponibles
python3 -c "
import cv2
print('SIFT disponible:', hasattr(cv2, 'SIFT_create'))
print('ORB disponible:', hasattr(cv2, 'ORB_create'))
"
```

### Memoria Insuficiente

**Síntomas**:
```
MemoryError: Unable to allocate array
cv2.error: Insufficient memory
```

**Diagnóstico**:
```bash
# Monitorear memoria durante procesamiento
watch -n 1 'free -h && ps aux | grep python | head -5'
```

**Solución**:
```python
# Configurar límites de memoria
import resource

# Limitar memoria a 2GB
resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, -1))

# Procesar imágenes en lotes más pequeños
def process_batch(images, batch_size=5):
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        yield process_images(batch)
```

### Timeout de Procesamiento

**Síntomas**:
```
TimeoutError: Processing timeout after 300 seconds
```

**Solución**:
```python
# Aumentar timeout en configuración
{
    "processing": {
        "timeout": 600,  # 10 minutos
        "max_image_size": 10485760,  # 10MB
        "enable_parallel": true
    }
}

# Implementar procesamiento asíncrono
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_image_async(image_path):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, process_image, image_path
        )
    return result
```

---

## Problemas de Rendimiento

### Alto Uso de CPU

**Síntomas**:
- CPU al 100% constantemente
- Sistema lento
- Timeouts frecuentes

**Diagnóstico**:
```bash
# Monitorear procesos
htop
ps aux --sort=-%cpu | head -10

# Profiling de Python
python3 -m cProfile -o profile.stats main.py
python3 -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(10)
"
```

**Solución**:
```python
# Optimizar algoritmos
def optimize_lbp_processing():
    # Usar NumPy vectorizado
    import numpy as np
    
    # En lugar de loops anidados
    def lbp_optimized(image, radius=1, n_points=8):
        # Implementación vectorizada
        pass

# Configurar paralelismo
{
    "processing": {
        "parallel_workers": 4,  # Número de cores - 1
        "chunk_size": 100,
        "use_multiprocessing": true
    }
}
```

### Alto Uso de Memoria

**Síntomas**:
- Memoria RAM al 90%+
- Swap activo
- OOM Killer activado

**Diagnóstico**:
```bash
# Monitorear memoria
free -h
cat /proc/meminfo
ps aux --sort=-%mem | head -10

# Verificar memory leaks
valgrind --tool=memcheck python3 main.py
```

**Solución**:
```python
# Gestión de memoria
import gc
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > 1000:  # 1GB
        gc.collect()
        print(f"Memoria liberada: {memory_mb:.1f}MB")

# Procesamiento por chunks
def process_large_dataset(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        process_chunk(chunk)
        del chunk  # Liberar memoria explícitamente
        gc.collect()
```

### Disco Lento

**Síntomas**:
- I/O wait alto
- Operaciones de archivo lentas
- Base de datos lenta

**Diagnóstico**:
```bash
# Monitorear I/O
iostat -x 1
iotop

# Verificar espacio en disco
df -h
du -sh /var/lib/sigec-balistica/*

# Verificar salud del disco
sudo smartctl -a /dev/sda
```

**Solución**:
```bash
# Optimizar PostgreSQL
# En postgresql.conf:
checkpoint_segments = 32
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Mover datos a SSD si es posible
sudo rsync -av /var/lib/sigec-balistica/ /ssd/sigec-balistica/
sudo ln -sf /ssd/sigec-balistica /var/lib/sigec-balistica

# Configurar cache
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
```

---

## Problemas de Interfaz

### GUI No Responde

**Síntomas**:
- Ventana congelada
- Botones no funcionan
- Aplicación no responde

**Diagnóstico**:
```bash
# Verificar proceso GUI
ps aux | grep python | grep gui
xwininfo -tree -root | grep -i sigec

# Verificar X11
echo $DISPLAY
xdpyinfo | head -10
```

**Solución**:
```python
# Implementar threading para GUI
import threading
from PyQt5.QtCore import QThread, pyqtSignal

class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    
    def run(self):
        # Procesamiento en hilo separado
        for i in range(100):
            # Trabajo pesado
            self.progress.emit(i)
        self.finished.emit(result)

# En la GUI principal
def start_processing(self):
    self.thread = ProcessingThread()
    self.thread.progress.connect(self.update_progress)
    self.thread.finished.connect(self.on_finished)
    self.thread.start()
```

### Error de Display

**Síntomas**:
```
qt.qpa.xcb: could not connect to display
DISPLAY variable not set
```

**Solución**:
```bash
# Para SSH con X11 forwarding
ssh -X usuario@servidor

# Configurar DISPLAY
export DISPLAY=:0.0

# Usar Xvfb para headless
sudo apt install xvfb
xvfb-run -a python3 gui_app.py

# Configurar VNC
sudo apt install tightvncserver
vncserver :1 -geometry 1024x768 -depth 24
```

### Problemas de Renderizado

**Síntomas**:
- Imágenes no se muestran
- Gráficos corruptos
- Colores incorrectos

**Solución**:
```python
# Verificar formato de imagen
def fix_image_display(image_path):
    import cv2
    from PyQt5.QtGui import QPixmap, QImage
    
    # Leer imagen
    img = cv2.imread(image_path)
    
    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crear QImage
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    return QPixmap.fromImage(qt_image)
```

---

## Problemas de Red

### API No Responde

**Síntomas**:
```
requests.exceptions.ConnectionError: Connection refused
curl: (7) Failed to connect to localhost port 8000
```

**Diagnóstico**:
```bash
# Verificar puerto
netstat -tlnp | grep :8000
ss -tlnp | grep :8000

# Probar conectividad
curl -v http://localhost:8000/health
telnet localhost 8000

# Verificar firewall
sudo ufw status
sudo iptables -L
```

**Solución**:
```bash
# Abrir puerto en firewall
sudo ufw allow 8000
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT

# Verificar configuración de bind
# En config.json:
{
    "server": {
        "host": "0.0.0.0",  # No solo localhost
        "port": 8000
    }
}

# Reiniciar servicio
sudo systemctl restart sigec-balistica
```

### Timeout de Red

**Síntomas**:
```
requests.exceptions.ReadTimeout: Read timed out
```

**Solución**:
```python
# Configurar timeouts apropiados
import requests

session = requests.Session()
session.timeout = (30, 300)  # (connect, read)

# Implementar retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

### SSL/TLS Problemas

**Síntomas**:
```
SSL: CERTIFICATE_VERIFY_FAILED
requests.exceptions.SSLError
```

**Solución**:
```bash
# Verificar certificados
openssl s_client -connect api.sigec-balistica.com:443
curl -vI https://api.sigec-balistica.com

# Actualizar certificados
sudo apt update && sudo apt install ca-certificates
sudo update-ca-certificates

# Configurar certificado personalizado
sudo cp custom-cert.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

---

## Problemas de Seguridad

### Acceso No Autorizado

**Síntomas**:
```
HTTP 401 Unauthorized
HTTP 403 Forbidden
```

**Diagnóstico**:
```bash
# Verificar logs de autenticación
tail -f /var/log/sigec-balistica/auth.log
grep "401\|403" /var/log/nginx/access.log

# Verificar tokens
python3 -c "
import jwt
token = 'your_token_here'
try:
    decoded = jwt.decode(token, verify=False)
    print('Token válido:', decoded)
except:
    print('Token inválido')
"
```

**Solución**:
```python
# Renovar token
def refresh_token(refresh_token):
    response = requests.post('/api/v1/auth/refresh', {
        'refresh_token': refresh_token
    })
    return response.json()['access_token']

# Verificar permisos
def check_permissions(user_id, resource):
    permissions = get_user_permissions(user_id)
    return resource in permissions
```

### Vulnerabilidades de Seguridad

**Diagnóstico**:
```bash
# Escanear vulnerabilidades
pip-audit
safety check

# Verificar configuración
python3 -c "
import ssl
context = ssl.create_default_context()
print('TLS version:', context.protocol)
print('Ciphers:', context.get_ciphers()[:3])
"
```

**Solución**:
```bash
# Actualizar dependencias
pip3 install --upgrade -r requirements.txt

# Configurar HTTPS
# En nginx.conf:
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
}
```

---

## Herramientas de Diagnóstico

### Script de Diagnóstico Completo

```bash
#!/bin/bash
# diagnostic_tool.sh - Herramienta de diagnóstico completa

LOG_FILE="/tmp/sigec_diagnostic_$(date +%Y%m%d_%H%M%S).log"

echo "=== DIAGNÓSTICO SIGEC-BALISTICA ===" | tee $LOG_FILE
echo "Fecha: $(date)" | tee -a $LOG_FILE
echo "Usuario: $(whoami)" | tee -a $LOG_FILE
echo "Hostname: $(hostname)" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

# Información del sistema
echo "1. INFORMACIÓN DEL SISTEMA" | tee -a $LOG_FILE
echo "OS: $(lsb_release -d | cut -f2)" | tee -a $LOG_FILE
echo "Kernel: $(uname -r)" | tee -a $LOG_FILE
echo "Arquitectura: $(uname -m)" | tee -a $LOG_FILE
echo "Uptime: $(uptime)" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

# Recursos del sistema
echo "2. RECURSOS DEL SISTEMA" | tee -a $LOG_FILE
echo "CPU:" | tee -a $LOG_FILE
lscpu | grep -E "Model name|CPU\(s\)|Thread" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

echo "Memoria:" | tee -a $LOG_FILE
free -h | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

echo "Disco:" | tee -a $LOG_FILE
df -h | grep -E "/$|/var|/tmp" | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

# Estado de servicios
echo "3. ESTADO DE SERVICIOS" | tee -a $LOG_FILE
for service in sigec-balistica postgresql nginx; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service: activo" | tee -a $LOG_FILE
    else
        echo "❌ $service: inactivo" | tee -a $LOG_FILE
    fi
done
echo | tee -a $LOG_FILE

# Conectividad
echo "4. CONECTIVIDAD" | tee -a $LOG_FILE
for port in 8000 5432 80 443; do
    if netstat -tlnp | grep -q ":$port "; then
        echo "✅ Puerto $port: abierto" | tee -a $LOG_FILE
    else
        echo "❌ Puerto $port: cerrado" | tee -a $LOG_FILE
    fi
done
echo | tee -a $LOG_FILE

# Logs recientes
echo "5. LOGS RECIENTES" | tee -a $LOG_FILE
echo "Errores de aplicación:" | tee -a $LOG_FILE
if [ -f /var/log/sigec-balistica/error.log ]; then
    tail -n 10 /var/log/sigec-balistica/error.log | tee -a $LOG_FILE
else
    echo "No se encontró log de errores" | tee -a $LOG_FILE
fi
echo | tee -a $LOG_FILE

echo "Errores del sistema:" | tee -a $LOG_FILE
journalctl -u sigec-balistica --since "1 hour ago" --no-pager | tail -n 10 | tee -a $LOG_FILE
echo | tee -a $LOG_FILE

# Configuración
echo "6. CONFIGURACIÓN" | tee -a $LOG_FILE
if [ -f /etc/sigec-balistica/config.json ]; then
    echo "Archivo de configuración encontrado" | tee -a $LOG_FILE
    # No mostrar contenido por seguridad
else
    echo "❌ Archivo de configuración no encontrado" | tee -a $LOG_FILE
fi
echo | tee -a $LOG_FILE

echo "Diagnóstico completado. Log guardado en: $LOG_FILE"
```

### Monitor de Rendimiento

```python
#!/usr/bin/env python3
# performance_monitor.py

import psutil
import time
import json
from datetime import datetime

def monitor_system(duration=300, interval=5):
    """Monitor sistema por duración especificada"""
    
    data = []
    start_time = time.time()
    
    print(f"Monitoreando sistema por {duration} segundos...")
    
    while time.time() - start_time < duration:
        timestamp = datetime.now().isoformat()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memoria
        memory = psutil.virtual_memory()
        
        # Disco
        disk = psutil.disk_usage('/')
        
        # Red
        net_io = psutil.net_io_counters()
        
        # Procesos SIGeC
        sigec_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if 'sigec' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                sigec_processes.append(proc.info)
        
        record = {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            },
            'processes': sigec_processes
        }
        
        data.append(record)
        
        # Mostrar progreso
        elapsed = time.time() - start_time
        progress = (elapsed / duration) * 100
        print(f"\rProgreso: {progress:.1f}% - CPU: {cpu_percent:.1f}% - RAM: {memory.percent:.1f}%", end='')
        
        time.sleep(interval)
    
    print("\nMonitoreo completado.")
    
    # Guardar datos
    filename = f"performance_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Datos guardados en: {filename}")
    
    # Análisis básico
    analyze_performance(data)

def analyze_performance(data):
    """Analizar datos de rendimiento"""
    
    if not data:
        return
    
    cpu_values = [record['cpu']['percent'] for record in data]
    memory_values = [record['memory']['percent'] for record in data]
    
    print("\n=== ANÁLISIS DE RENDIMIENTO ===")
    print(f"CPU promedio: {sum(cpu_values)/len(cpu_values):.1f}%")
    print(f"CPU máximo: {max(cpu_values):.1f}%")
    print(f"Memoria promedio: {sum(memory_values)/len(memory_values):.1f}%")
    print(f"Memoria máximo: {max(memory_values):.1f}%")
    
    # Alertas
    if max(cpu_values) > 90:
        print("⚠️  ALERTA: CPU muy alto detectado")
    if max(memory_values) > 90:
        print("⚠️  ALERTA: Memoria muy alta detectada")

if __name__ == "__main__":
    import sys
    
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    monitor_system(duration, interval)
```

### Validador de Configuración

```python
#!/usr/bin/env python3
# config_validator.py

import json
import os
import sys
from pathlib import Path

def validate_config():
    """Validar configuración del sistema"""
    
    issues = []
    warnings = []
    
    print("=== VALIDADOR DE CONFIGURACIÓN ===\n")
    
    # 1. Verificar archivo de configuración
    config_path = "/etc/sigec-balistica/config.json"
    if not os.path.exists(config_path):
        issues.append(f"Archivo de configuración no encontrado: {config_path}")
        return issues, warnings
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(f"Error en JSON de configuración: {e}")
        return issues, warnings
    
    print("✅ Archivo de configuración válido")
    
    # 2. Verificar estructura de configuración
    required_sections = ['database', 'server', 'processing', 'logging']
    for section in required_sections:
        if section not in config:
            issues.append(f"Sección faltante en configuración: {section}")
    
    # 3. Verificar configuración de base de datos
    if 'database' in config:
        db_config = config['database']
        required_db_fields = ['host', 'port', 'name', 'user', 'password']
        for field in required_db_fields:
            if field not in db_config:
                issues.append(f"Campo faltante en database: {field}")
    
    # 4. Verificar directorios
    directories = [
        '/var/log/sigec-balistica',
        '/var/lib/sigec-balistica',
        '/tmp/sigec-balistica'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            issues.append(f"Directorio faltante: {directory}")
        elif not os.access(directory, os.W_OK):
            issues.append(f"Sin permisos de escritura: {directory}")
    
    # 5. Verificar dependencias de Python
    required_packages = [
        'opencv-python',
        'scikit-learn',
        'numpy',
        'psycopg2-binary',
        'flask',
        'PyQt5'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Paquetes Python faltantes: {', '.join(missing_packages)}")
    
    # 6. Verificar configuración de seguridad
    if 'security' in config:
        security = config['security']
        if security.get('debug', True):
            warnings.append("Modo debug habilitado en producción")
        if not security.get('https_only', False):
            warnings.append("HTTPS no está forzado")
    
    # 7. Verificar límites de recursos
    if 'processing' in config:
        processing = config['processing']
        max_memory = processing.get('max_memory_mb', 1024)
        if max_memory > 8192:
            warnings.append(f"Límite de memoria muy alto: {max_memory}MB")
    
    # Mostrar resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Problemas críticos: {len(issues)}")
    print(f"Advertencias: {len(warnings)}")
    
    if issues:
        print("\n❌ PROBLEMAS CRÍTICOS:")
        for issue in issues:
            print(f"  • {issue}")
    
    if warnings:
        print("\n⚠️  ADVERTENCIAS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not issues and not warnings:
        print("\n✅ Configuración válida")
    
    return issues, warnings

if __name__ == "__main__":
    issues, warnings = validate_config()
    sys.exit(1 if issues else 0)
```

---

## Logs y Monitoreo

### Configuración de Logs

```python
# logging_config.py
import logging
import logging.handlers
import os

def setup_logging():
    """Configurar sistema de logging"""
    
    # Crear directorios si no existen
    log_dir = "/var/log/sigec-balistica"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Logger principal
    logger = logging.getLogger('sigec')
    logger.setLevel(logging.INFO)
    
    # Handler para archivo principal
    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para errores
    error_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/error.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Handler para consola (desarrollo)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
```

### Análisis de Logs

```bash
#!/bin/bash
# log_analyzer.sh

LOG_DIR="/var/log/sigec-balistica"
REPORT_FILE="/tmp/log_analysis_$(date +%Y%m%d_%H%M%S).txt"

echo "=== ANÁLISIS DE LOGS SIGEC-BALISTICA ===" > $REPORT_FILE
echo "Fecha: $(date)" >> $REPORT_FILE
echo >> $REPORT_FILE

# Errores más frecuentes
echo "1. ERRORES MÁS FRECUENTES (últimas 24h):" >> $REPORT_FILE
find $LOG_DIR -name "*.log" -mtime -1 -exec grep -h "ERROR" {} \; | \
    cut -d'-' -f4- | sort | uniq -c | sort -nr | head -10 >> $REPORT_FILE
echo >> $REPORT_FILE

# Actividad por hora
echo "2. ACTIVIDAD POR HORA (hoy):" >> $REPORT_FILE
grep "$(date +%Y-%m-%d)" $LOG_DIR/app.log | \
    cut -d' ' -f2 | cut -d':' -f1 | sort | uniq -c >> $REPORT_FILE
echo >> $REPORT_FILE

# Usuarios más activos
echo "3. USUARIOS MÁS ACTIVOS:" >> $REPORT_FILE
grep "user:" $LOG_DIR/app.log | \
    sed 's/.*user:\([^,]*\).*/\1/' | sort | uniq -c | sort -nr | head -10 >> $REPORT_FILE
echo >> $REPORT_FILE

# Análisis de rendimiento
echo "4. ANÁLISIS DE RENDIMIENTO:" >> $REPORT_FILE
echo "Análisis completados hoy: $(grep -c "Analysis completed" $LOG_DIR/app.log)" >> $REPORT_FILE
echo "Tiempo promedio de análisis:" >> $REPORT_FILE
grep "processing_time" $LOG_DIR/app.log | \
    sed 's/.*processing_time:\([0-9.]*\).*/\1/' | \
    awk '{sum+=$1; count++} END {if(count>0) print sum/count "s"}' >> $REPORT_FILE
echo >> $REPORT_FILE

# Alertas de seguridad
echo "5. ALERTAS DE SEGURIDAD:" >> $REPORT_FILE
grep -E "(401|403|failed login|unauthorized)" $LOG_DIR/*.log | wc -l | \
    xargs echo "Intentos de acceso no autorizado:" >> $REPORT_FILE
echo >> $REPORT_FILE

echo "Análisis completado. Reporte guardado en: $REPORT_FILE"
cat $REPORT_FILE
```

### Monitoreo en Tiempo Real

```bash
#!/bin/bash
# realtime_monitor.sh

echo "=== MONITOR EN TIEMPO REAL SIGEC-BALISTICA ==="
echo "Presione Ctrl+C para salir"
echo

# Función para mostrar estado
show_status() {
    clear
    echo "=== ESTADO DEL SISTEMA $(date) ==="
    echo
    
    # Estado de servicios
    echo "SERVICIOS:"
    systemctl is-active sigec-balistica | sed 's/active/✅ SIGeC: Activo/' | sed 's/inactive/❌ SIGeC: Inactivo/'
    systemctl is-active postgresql | sed 's/active/✅ PostgreSQL: Activo/' | sed 's/inactive/❌ PostgreSQL: Inactivo/'
    echo
    
    # Recursos
    echo "RECURSOS:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "RAM: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
    echo "Disco: $(df -h / | awk 'NR==2{printf "%s", $5}')"
    echo
    
    # Actividad reciente
    echo "ACTIVIDAD RECIENTE:"
    tail -n 5 /var/log/sigec-balistica/app.log | cut -d'-' -f3- | sed 's/^/ /'
    echo
    
    # Conexiones activas
    echo "CONEXIONES:"
    netstat -tn | grep :8000 | wc -l | xargs echo "API:"
    netstat -tn | grep :5432 | wc -l | xargs echo "Database:"
}

# Loop principal
while true; do
    show_status
    sleep 5
done
```

---

## Contacto de Soporte

### Información de Contacto

**Soporte Técnico**:
- 📧 Email: soporte@sigec-balistica.com
- 📞 Teléfono: +1-800-SIGEC-HELP
- 🌐 Web: https://support.sigec-balistica.com
- 💬 Chat: Disponible 24/7 en el portal web

**Soporte de Emergencia**:
- 📞 Teléfono: +1-800-SIGEC-911
- 📧 Email: emergency@sigec-balistica.com

### Información a Incluir en Reportes

Cuando contacte al soporte, incluya:

1. **Información del Sistema**:
   ```bash
   uname -a
   lsb_release -a
   python3 --version
   ```

2. **Logs Relevantes**:
   ```bash
   tail -n 50 /var/log/sigec-balistica/error.log
   journalctl -u sigec-balistica --since "1 hour ago"
   ```

3. **Configuración** (sin contraseñas):
   ```bash
   cat /etc/sigec-balistica/config.json | jq 'del(.database.password)'
   ```

4. **Descripción del Problema**:
   - Qué estaba haciendo cuando ocurrió
   - Mensaje de error exacto
   - Pasos para reproducir
   - Frecuencia del problema

### Niveles de Soporte

| Nivel | Descripción | Tiempo de Respuesta |
|-------|-------------|-------------------|
| **Crítico** | Sistema no funciona | 1 hora |
| **Alto** | Funcionalidad importante afectada | 4 horas |
| **Medio** | Problema menor | 24 horas |
| **Bajo** | Consulta general | 72 horas |

### Recursos Adicionales

- 📚 **Documentación**: https://docs.sigec-balistica.com
- 🎓 **Tutoriales**: https://learn.sigec-balistica.com
- 💬 **Foro Comunitario**: https://community.sigec-balistica.com
- 📺 **Videos**: https://youtube.com/sigec-balistica
- 📖 **Base de Conocimiento**: https://kb.sigec-balistica.com

---

*Guía de Solución de Problemas - SIGeC-Balistica v1.0*  
*Última actualización: Enero 2024*  
*© 2024 SIGeC-Balistica. Todos los derechos reservados.*