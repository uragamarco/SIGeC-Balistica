---
title: Procedimientos de Mantenimiento
system: SIGeC-Balisticar
language: es-ES
version: current
last_updated: 2025-10-16
status: active
audience: devops, dba, seguridad
toc: true
tags:
  - mantenimiento
  - respaldos
  - rendimiento
  - seguridad
---

# SIGeC-Balisticar - Procedimientos de Mantenimiento

## Tabla de Contenidos
1. [Mantenimiento Preventivo](#mantenimiento-preventivo)
2. [Respaldos y Recuperación](#respaldos-y-recuperación)
3. [Actualizaciones del Sistema](#actualizaciones-del-sistema)
4. [Monitoreo y Alertas](#monitoreo-y-alertas)
5. [Optimización de Rendimiento](#optimización-de-rendimiento)
6. [Gestión de Logs](#gestión-de-logs)
7. [Seguridad y Auditoría](#seguridad-y-auditoría)
8. [Procedimientos de Emergencia](#procedimientos-de-emergencia)
9. [Mantenimiento de Base de Datos](#mantenimiento-de-base-de-datos)
10. [Documentación y Reportes](#documentación-y-reportes)

---

## Mantenimiento Preventivo

### Cronograma de Mantenimiento

| Frecuencia | Tarea | Responsable | Duración Estimada |
|------------|-------|-------------|-------------------|
| **Diario** | Verificar logs de error | Administrador | 15 min |
| **Diario** | Monitorear recursos del sistema | Administrador | 10 min |
| **Semanal** | Limpiar archivos temporales | Administrador | 30 min |
| **Semanal** | Verificar respaldos | Administrador | 20 min |
| **Mensual** | Actualizar dependencias | DevOps | 2 horas |
| **Mensual** | Optimizar base de datos | DBA | 1 hora |
| **Trimestral** | Auditoría de seguridad | Seguridad | 4 horas |
| **Semestral** | Revisión de arquitectura | Arquitecto | 8 horas |

### Lista de Verificación Diaria

```bash
#!/bin/bash
# daily_maintenance.sh - Mantenimiento diario automatizado

LOG_FILE="/var/log/sigec-balistica/maintenance_$(date +%Y%m%d).log"

echo "=== MANTENIMIENTO DIARIO $(date) ===" | tee -a $LOG_FILE

# 1. Verificar estado de servicios
echo "1. Verificando servicios..." | tee -a $LOG_FILE
for service in sigec-balistica postgresql redis nginx; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service: Activo" | tee -a $LOG_FILE
    else
        echo "❌ $service: Inactivo - ALERTA" | tee -a $LOG_FILE
        # Intentar reiniciar
        systemctl restart $service
        sleep 5
        if systemctl is-active --quiet $service; then
            echo "✅ $service: Reiniciado exitosamente" | tee -a $LOG_FILE
        else
            echo "❌ $service: Fallo al reiniciar - CRÍTICO" | tee -a $LOG_FILE
            # Enviar alerta
            send_alert "Servicio $service no pudo reiniciarse"
        fi
    fi
done

# 2. Verificar recursos del sistema
echo "2. Verificando recursos..." | tee -a $LOG_FILE

# CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "⚠️  CPU alto: ${CPU_USAGE}%" | tee -a $LOG_FILE
    send_alert "CPU usage alto: ${CPU_USAGE}%"
else
    echo "✅ CPU: ${CPU_USAGE}%" | tee -a $LOG_FILE
fi

# Memoria
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
    echo "⚠️  Memoria alta: ${MEMORY_USAGE}%" | tee -a $LOG_FILE
    send_alert "Memoria usage alta: ${MEMORY_USAGE}%"
else
    echo "✅ Memoria: ${MEMORY_USAGE}%" | tee -a $LOG_FILE
fi

# Disco
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "⚠️  Disco lleno: ${DISK_USAGE}%" | tee -a $LOG_FILE
    send_alert "Disco usage alto: ${DISK_USAGE}%"
else
    echo "✅ Disco: ${DISK_USAGE}%" | tee -a $LOG_FILE
fi

# 3. Verificar conectividad
echo "3. Verificando conectividad..." | tee -a $LOG_FILE

# API
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✅ API: Responde" | tee -a $LOG_FILE
else
    echo "❌ API: No responde" | tee -a $LOG_FILE
    send_alert "API no responde"
fi

# Base de datos
if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Disponible" | tee -a $LOG_FILE
else
    echo "❌ PostgreSQL: No disponible" | tee -a $LOG_FILE
    send_alert "PostgreSQL no disponible"
fi

# 4. Verificar logs de error
echo "4. Verificando logs de error..." | tee -a $LOG_FILE
ERROR_COUNT=$(grep -c "ERROR" /var/log/sigec-balistica/error.log 2>/dev/null || echo "0")
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "⚠️  Errores detectados: $ERROR_COUNT" | tee -a $LOG_FILE
    echo "Últimos errores:" | tee -a $LOG_FILE
    tail -n 5 /var/log/sigec-balistica/error.log | tee -a $LOG_FILE
    send_alert "Múltiples errores detectados: $ERROR_COUNT"
else
    echo "✅ Errores: $ERROR_COUNT (normal)" | tee -a $LOG_FILE
fi

# 5. Limpiar archivos temporales
echo "5. Limpiando archivos temporales..." | tee -a $LOG_FILE
find /tmp/sigec-balistica -type f -mtime +1 -delete 2>/dev/null
find /var/log/sigec-balistica -name "*.log.*" -mtime +30 -delete 2>/dev/null
echo "✅ Limpieza completada" | tee -a $LOG_FILE

echo "=== MANTENIMIENTO COMPLETADO ===" | tee -a $LOG_FILE

# Función para enviar alertas
send_alert() {
    local message="$1"
    echo "ALERTA: $message" | mail -s "SIGeC-Balistica Alert" admin@company.com
    # También se puede integrar con Slack, Teams, etc.
}
```

### Lista de Verificación Semanal

```bash
#!/bin/bash
# weekly_maintenance.sh - Mantenimiento semanal

LOG_FILE="/var/log/sigec-balistica/weekly_maintenance_$(date +%Y%m%d).log"

echo "=== MANTENIMIENTO SEMANAL $(date) ===" | tee -a $LOG_FILE

# 1. Verificar respaldos
echo "1. Verificando respaldos..." | tee -a $LOG_FILE
BACKUP_DIR="/backup/sigec-balistica"
LATEST_BACKUP=$(ls -t $BACKUP_DIR/*.sql 2>/dev/null | head -n1)

if [ -n "$LATEST_BACKUP" ]; then
    BACKUP_AGE=$(find "$LATEST_BACKUP" -mtime +1 | wc -l)
    if [ "$BACKUP_AGE" -eq 0 ]; then
        echo "✅ Backup reciente encontrado: $(basename $LATEST_BACKUP)" | tee -a $LOG_FILE
    else
        echo "⚠️  Backup antiguo: $(basename $LATEST_BACKUP)" | tee -a $LOG_FILE
        # Crear nuevo backup
        create_backup
    fi
else
    echo "❌ No se encontraron backups" | tee -a $LOG_FILE
    create_backup
fi

# 2. Análisis de logs
echo "2. Analizando logs..." | tee -a $LOG_FILE
python3 /opt/sigec-balistica/scripts/log_analyzer.py --week | tee -a $LOG_FILE

# 3. Verificar integridad de archivos
echo "3. Verificando integridad de archivos..." | tee -a $LOG_FILE
find /var/lib/sigec-balistica -name "*.jpg" -o -name "*.png" | while read file; do
    if ! file "$file" | grep -q "image"; then
        echo "⚠️  Archivo corrupto: $file" | tee -a $LOG_FILE
    fi
done

# 4. Optimizar base de datos
echo "4. Optimizando base de datos..." | tee -a $LOG_FILE
psql -d sigec_balistica -c "VACUUM ANALYZE;" | tee -a $LOG_FILE

# 5. Verificar certificados SSL
echo "5. Verificando certificados SSL..." | tee -a $LOG_FILE
CERT_FILE="/etc/ssl/certs/sigec.crt"
if [ -f "$CERT_FILE" ]; then
    EXPIRY_DATE=$(openssl x509 -in "$CERT_FILE" -noout -enddate | cut -d= -f2)
    EXPIRY_TIMESTAMP=$(date -d "$EXPIRY_DATE" +%s)
    CURRENT_TIMESTAMP=$(date +%s)
    DAYS_LEFT=$(( (EXPIRY_TIMESTAMP - CURRENT_TIMESTAMP) / 86400 ))
    
    if [ "$DAYS_LEFT" -lt 30 ]; then
        echo "⚠️  Certificado SSL expira en $DAYS_LEFT días" | tee -a $LOG_FILE
        send_alert "Certificado SSL expira pronto: $DAYS_LEFT días"
    else
        echo "✅ Certificado SSL válido por $DAYS_LEFT días" | tee -a $LOG_FILE
    fi
fi

echo "=== MANTENIMIENTO SEMANAL COMPLETADO ===" | tee -a $LOG_FILE

create_backup() {
    echo "Creando backup..." | tee -a $LOG_FILE
    BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql"
    pg_dump sigec_balistica > "$BACKUP_FILE"
    if [ $? -eq 0 ]; then
        echo "✅ Backup creado: $(basename $BACKUP_FILE)" | tee -a $LOG_FILE
        # Comprimir backup
        gzip "$BACKUP_FILE"
    else
        echo "❌ Error creando backup" | tee -a $LOG_FILE
        send_alert "Error creando backup de base de datos"
    fi
}
```

---

## Respaldos y Recuperación

### Estrategia de Respaldos

#### Tipos de Respaldo

1. **Respaldo Completo** (Semanal)
   - Base de datos completa
   - Archivos de configuración
   - Imágenes y documentos
   - Logs importantes

2. **Respaldo Incremental** (Diario)
   - Solo cambios desde el último respaldo
   - Transacciones de base de datos
   - Nuevos archivos

3. **Respaldo de Configuración** (Antes de cambios)
   - Archivos de configuración
   - Scripts de despliegue
   - Certificados SSL

### Script de Respaldo Automatizado

```bash
#!/bin/bash
# backup_system.sh - Sistema de respaldos automatizado

# Configuración
BACKUP_BASE_DIR="/backup/sigec-balistica"
RETENTION_DAYS=30
RETENTION_WEEKS=12
RETENTION_MONTHS=12

# Crear directorios
mkdir -p "$BACKUP_BASE_DIR"/{daily,weekly,monthly}

# Función de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_BASE_DIR/backup.log"
}

# Respaldo de base de datos
backup_database() {
    local backup_type="$1"
    local backup_dir="$BACKUP_BASE_DIR/$backup_type"
    local backup_file="$backup_dir/db_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    log "Iniciando respaldo de base de datos ($backup_type)"
    
    # Crear respaldo
    pg_dump -h localhost -U sigec_user sigec_balistica > "$backup_file"
    
    if [ $? -eq 0 ]; then
        # Comprimir
        gzip "$backup_file"
        log "✅ Respaldo de BD completado: $(basename $backup_file.gz)"
        
        # Verificar integridad
        if gunzip -t "$backup_file.gz"; then
            log "✅ Integridad del respaldo verificada"
        else
            log "❌ Error en integridad del respaldo"
            return 1
        fi
    else
        log "❌ Error creando respaldo de BD"
        return 1
    fi
}

# Respaldo de archivos
backup_files() {
    local backup_type="$1"
    local backup_dir="$BACKUP_BASE_DIR/$backup_type"
    local backup_file="$backup_dir/files_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    log "Iniciando respaldo de archivos ($backup_type)"
    
    # Directorios a respaldar
    DIRS_TO_BACKUP=(
        "/etc/sigec-balistica"
        "/var/lib/sigec-balistica"
        "/opt/sigec-balistica/config"
        "/etc/ssl/certs/sigec*"
        "/etc/nginx/sites-available/sigec*"
    )
    
    # Crear archivo tar
    tar -czf "$backup_file" "${DIRS_TO_BACKUP[@]}" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        log "✅ Respaldo de archivos completado: $(basename $backup_file)"
    else
        log "❌ Error creando respaldo de archivos"
        return 1
    fi
}

# Limpiar respaldos antiguos
cleanup_old_backups() {
    log "Limpiando respaldos antiguos"
    
    # Respaldos diarios (mantener 30 días)
    find "$BACKUP_BASE_DIR/daily" -name "*.gz" -mtime +$RETENTION_DAYS -delete
    
    # Respaldos semanales (mantener 12 semanas)
    find "$BACKUP_BASE_DIR/weekly" -name "*.gz" -mtime +$((RETENTION_WEEKS * 7)) -delete
    
    # Respaldos mensuales (mantener 12 meses)
    find "$BACKUP_BASE_DIR/monthly" -name "*.gz" -mtime +$((RETENTION_MONTHS * 30)) -delete
    
    log "✅ Limpieza de respaldos completada"
}

# Verificar espacio en disco
check_disk_space() {
    local required_space_gb=5
    local available_space=$(df -BG "$BACKUP_BASE_DIR" | awk 'NR==2{print $4}' | sed 's/G//')
    
    if [ "$available_space" -lt "$required_space_gb" ]; then
        log "⚠️  Espacio insuficiente: ${available_space}GB disponible, ${required_space_gb}GB requerido"
        return 1
    fi
    
    log "✅ Espacio suficiente: ${available_space}GB disponible"
    return 0
}

# Función principal
main() {
    local backup_type="${1:-daily}"
    
    log "=== INICIANDO RESPALDO $backup_type ==="
    
    # Verificar espacio
    if ! check_disk_space; then
        log "❌ Respaldo cancelado por falta de espacio"
        exit 1
    fi
    
    # Realizar respaldos
    backup_database "$backup_type"
    backup_files "$backup_type"
    
    # Limpiar respaldos antiguos
    cleanup_old_backups
    
    # Enviar notificación
    send_backup_notification "$backup_type"
    
    log "=== RESPALDO $backup_type COMPLETADO ==="
}

# Enviar notificación
send_backup_notification() {
    local backup_type="$1"
    local backup_size=$(du -sh "$BACKUP_BASE_DIR/$backup_type" | cut -f1)
    
    echo "Respaldo $backup_type completado exitosamente.
Tamaño: $backup_size
Fecha: $(date)
Ubicación: $BACKUP_BASE_DIR/$backup_type" | \
    mail -s "SIGeC-Balistica: Respaldo $backup_type completado" admin@company.com
}

# Ejecutar
main "$@"
```

### Procedimiento de Recuperación

```bash
#!/bin/bash
# restore_system.sh - Procedimiento de recuperación

# Configuración
BACKUP_BASE_DIR="/backup/sigec-balistica"
RESTORE_LOG="/var/log/sigec-balistica/restore_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESTORE_LOG"
}

# Listar respaldos disponibles
list_backups() {
    echo "Respaldos disponibles:"
    echo "====================="
    
    for type in daily weekly monthly; do
        echo "$type:"
        ls -la "$BACKUP_BASE_DIR/$type"/*.gz 2>/dev/null | \
            awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}'
        echo
    done
}

# Restaurar base de datos
restore_database() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        log "❌ Archivo de respaldo no encontrado: $backup_file"
        return 1
    fi
    
    log "Iniciando restauración de base de datos desde: $(basename $backup_file)"
    
    # Detener aplicación
    log "Deteniendo aplicación..."
    systemctl stop sigec-balistica
    
    # Crear respaldo de seguridad de la BD actual
    log "Creando respaldo de seguridad de BD actual..."
    pg_dump sigec_balistica > "/tmp/current_db_backup_$(date +%Y%m%d_%H%M%S).sql"
    
    # Restaurar base de datos
    log "Restaurando base de datos..."
    
    # Descomprimir si es necesario
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | psql -d sigec_balistica
    else
        psql -d sigec_balistica < "$backup_file"
    fi
    
    if [ $? -eq 0 ]; then
        log "✅ Base de datos restaurada exitosamente"
    else
        log "❌ Error restaurando base de datos"
        return 1
    fi
    
    # Reiniciar aplicación
    log "Reiniciando aplicación..."
    systemctl start sigec-balistica
    
    # Verificar funcionamiento
    sleep 10
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "✅ Aplicación funcionando correctamente"
    else
        log "⚠️  Aplicación no responde, verificar manualmente"
    fi
}

# Restaurar archivos
restore_files() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        log "❌ Archivo de respaldo no encontrado: $backup_file"
        return 1
    fi
    
    log "Iniciando restauración de archivos desde: $(basename $backup_file)"
    
    # Crear respaldo de archivos actuales
    log "Creando respaldo de archivos actuales..."
    tar -czf "/tmp/current_files_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
        /etc/sigec-balistica \
        /var/lib/sigec-balistica \
        /opt/sigec-balistica/config 2>/dev/null
    
    # Restaurar archivos
    log "Restaurando archivos..."
    tar -xzf "$backup_file" -C / --overwrite
    
    if [ $? -eq 0 ]; then
        log "✅ Archivos restaurados exitosamente"
        
        # Verificar permisos
        chown -R sigec:sigec /var/lib/sigec-balistica
        chown -R sigec:sigec /etc/sigec-balistica
        chmod 750 /etc/sigec-balistica
        
        log "✅ Permisos corregidos"
    else
        log "❌ Error restaurando archivos"
        return 1
    fi
}

# Restauración completa
full_restore() {
    local db_backup="$1"
    local files_backup="$2"
    
    log "=== INICIANDO RESTAURACIÓN COMPLETA ==="
    
    # Confirmar acción
    echo "⚠️  ADVERTENCIA: Esta operación sobrescribirá los datos actuales."
    echo "¿Está seguro de continuar? (escriba 'CONFIRMAR' para continuar)"
    read -r confirmation
    
    if [ "$confirmation" != "CONFIRMAR" ]; then
        log "Restauración cancelada por el usuario"
        return 1
    fi
    
    # Restaurar archivos primero
    if [ -n "$files_backup" ]; then
        restore_files "$files_backup"
    fi
    
    # Restaurar base de datos
    if [ -n "$db_backup" ]; then
        restore_database "$db_backup"
    fi
    
    log "=== RESTAURACIÓN COMPLETA FINALIZADA ==="
}

# Función principal
main() {
    case "$1" in
        "list")
            list_backups
            ;;
        "restore-db")
            restore_database "$2"
            ;;
        "restore-files")
            restore_files "$2"
            ;;
        "full-restore")
            full_restore "$2" "$3"
            ;;
        *)
            echo "Uso: $0 {list|restore-db|restore-files|full-restore} [archivo_respaldo] [archivo_archivos]"
            echo
            echo "Ejemplos:"
            echo "  $0 list"
            echo "  $0 restore-db /backup/sigec-balistica/daily/db_backup_20240115_120000.sql.gz"
            echo "  $0 restore-files /backup/sigec-balistica/daily/files_backup_20240115_120000.tar.gz"
            echo "  $0 full-restore db_backup.sql.gz files_backup.tar.gz"
            exit 1
            ;;
    esac
}

main "$@"
```

---

## Actualizaciones del Sistema

### Procedimiento de Actualización

```bash
#!/bin/bash
# update_system.sh - Procedimiento de actualización

# Configuración
APP_DIR="/opt/sigec-balistica"
BACKUP_DIR="/backup/sigec-balistica/updates"
UPDATE_LOG="/var/log/sigec-balistica/update_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$UPDATE_LOG"
}

# Pre-actualización
pre_update_checks() {
    log "=== VERIFICACIONES PRE-ACTUALIZACIÓN ==="
    
    # Verificar espacio en disco
    local available_space=$(df -BG "$APP_DIR" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 2 ]; then
        log "❌ Espacio insuficiente: ${available_space}GB disponible"
        return 1
    fi
    log "✅ Espacio suficiente: ${available_space}GB"
    
    # Verificar servicios
    for service in sigec-balistica postgresql redis; do
        if ! systemctl is-active --quiet $service; then
            log "❌ Servicio $service no está activo"
            return 1
        fi
    done
    log "✅ Todos los servicios están activos"
    
    # Verificar conectividad
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "❌ Aplicación no responde"
        return 1
    fi
    log "✅ Aplicación responde correctamente"
    
    return 0
}

# Crear respaldo pre-actualización
create_pre_update_backup() {
    log "Creando respaldo pre-actualización..."
    
    mkdir -p "$BACKUP_DIR"
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Respaldo de base de datos
    pg_dump sigec_balistica > "$BACKUP_DIR/pre_update_db_$backup_timestamp.sql"
    gzip "$BACKUP_DIR/pre_update_db_$backup_timestamp.sql"
    
    # Respaldo de archivos
    tar -czf "$BACKUP_DIR/pre_update_files_$backup_timestamp.tar.gz" \
        "$APP_DIR" \
        /etc/sigec-balistica \
        /var/lib/sigec-balistica
    
    log "✅ Respaldo pre-actualización completado"
}

# Actualizar dependencias del sistema
update_system_dependencies() {
    log "Actualizando dependencias del sistema..."
    
    apt update
    apt upgrade -y
    
    # Instalar nuevas dependencias si es necesario
    apt install -y python3-dev libopencv-dev
    
    log "✅ Dependencias del sistema actualizadas"
}

# Actualizar código de la aplicación
update_application_code() {
    log "Actualizando código de la aplicación..."
    
    cd "$APP_DIR"
    
    # Obtener versión actual
    local current_version=$(git describe --tags --always)
    log "Versión actual: $current_version"
    
    # Actualizar código
    git fetch origin
    git checkout main
    git pull origin main
    
    # Obtener nueva versión
    local new_version=$(git describe --tags --always)
    log "Nueva versión: $new_version"
    
    if [ "$current_version" = "$new_version" ]; then
        log "ℹ️  No hay actualizaciones de código disponibles"
        return 0
    fi
    
    log "✅ Código actualizado de $current_version a $new_version"
}

# Actualizar dependencias de Python
update_python_dependencies() {
    log "Actualizando dependencias de Python..."
    
    cd "$APP_DIR"
    
    # Activar entorno virtual
    source venv/bin/activate
    
    # Actualizar pip
    pip install --upgrade pip
    
    # Instalar/actualizar dependencias
    pip install -r requirements.txt --upgrade
    
    # Verificar dependencias
    pip check
    
    if [ $? -eq 0 ]; then
        log "✅ Dependencias de Python actualizadas"
    else
        log "⚠️  Conflictos en dependencias detectados"
        pip list --outdated | tee -a "$UPDATE_LOG"
    fi
}

# Ejecutar migraciones de base de datos
run_database_migrations() {
    log "Ejecutando migraciones de base de datos..."
    
    cd "$APP_DIR"
    source venv/bin/activate
    
    # Verificar migraciones pendientes
    python manage.py showmigrations --plan | grep -q "\[ \]"
    
    if [ $? -eq 0 ]; then
        log "Migraciones pendientes encontradas, ejecutando..."
        python manage.py migrate
        
        if [ $? -eq 0 ]; then
            log "✅ Migraciones ejecutadas exitosamente"
        else
            log "❌ Error ejecutando migraciones"
            return 1
        fi
    else
        log "ℹ️  No hay migraciones pendientes"
    fi
}

# Actualizar archivos estáticos
update_static_files() {
    log "Actualizando archivos estáticos..."
    
    cd "$APP_DIR"
    source venv/bin/activate
    
    # Recopilar archivos estáticos
    python manage.py collectstatic --noinput
    
    # Comprimir archivos CSS/JS si está configurado
    if command -v uglifyjs &> /dev/null; then
        find static/js -name "*.js" -not -name "*.min.js" -exec uglifyjs {} -o {}.min.js \;
    fi
    
    log "✅ Archivos estáticos actualizados"
}

# Reiniciar servicios
restart_services() {
    log "Reiniciando servicios..."
    
    # Reiniciar aplicación
    systemctl restart sigec-balistica
    sleep 5
    
    # Verificar que la aplicación esté funcionando
    if systemctl is-active --quiet sigec-balistica; then
        log "✅ Servicio sigec-balistica reiniciado"
    else
        log "❌ Error reiniciando sigec-balistica"
        return 1
    fi
    
    # Reiniciar nginx si es necesario
    nginx -t
    if [ $? -eq 0 ]; then
        systemctl reload nginx
        log "✅ Nginx recargado"
    else
        log "❌ Error en configuración de nginx"
        return 1
    fi
}

# Verificaciones post-actualización
post_update_checks() {
    log "=== VERIFICACIONES POST-ACTUALIZACIÓN ==="
    
    # Esperar a que la aplicación esté lista
    sleep 10
    
    # Verificar salud de la aplicación
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log "✅ Aplicación responde correctamente (intento $attempt)"
            break
        else
            log "⚠️  Aplicación no responde (intento $attempt/$max_attempts)"
            if [ $attempt -eq $max_attempts ]; then
                log "❌ Aplicación no responde después de $max_attempts intentos"
                return 1
            fi
            sleep 10
            ((attempt++))
        fi
    done
    
    # Verificar funcionalidades básicas
    log "Verificando funcionalidades básicas..."
    
    # Test de análisis básico
    if curl -f -X POST http://localhost:8000/api/v1/test/basic > /dev/null 2>&1; then
        log "✅ Funcionalidad básica verificada"
    else
        log "⚠️  Funcionalidad básica no responde"
    fi
    
    # Verificar base de datos
    if psql -d sigec_balistica -c "SELECT COUNT(*) FROM samples;" > /dev/null 2>&1; then
        log "✅ Base de datos accesible"
    else
        log "❌ Error accediendo a base de datos"
        return 1
    fi
    
    return 0
}

# Rollback en caso de error
rollback_update() {
    log "=== INICIANDO ROLLBACK ==="
    
    # Detener servicios
    systemctl stop sigec-balistica
    
    # Restaurar código
    cd "$APP_DIR"
    git checkout HEAD~1
    
    # Restaurar base de datos
    local latest_backup=$(ls -t "$BACKUP_DIR"/pre_update_db_*.sql.gz 2>/dev/null | head -n1)
    if [ -n "$latest_backup" ]; then
        log "Restaurando base de datos desde: $(basename $latest_backup)"
        gunzip -c "$latest_backup" | psql -d sigec_balistica
    fi
    
    # Restaurar archivos
    local latest_files_backup=$(ls -t "$BACKUP_DIR"/pre_update_files_*.tar.gz 2>/dev/null | head -n1)
    if [ -n "$latest_files_backup" ]; then
        log "Restaurando archivos desde: $(basename $latest_files_backup)"
        tar -xzf "$latest_files_backup" -C / --overwrite
    fi
    
    # Reiniciar servicios
    systemctl start sigec-balistica
    
    log "✅ Rollback completado"
}

# Función principal
main() {
    local skip_backup=false
    local force_update=false
    
    # Procesar argumentos
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-backup)
                skip_backup=true
                shift
                ;;
            --force)
                force_update=true
                shift
                ;;
            --rollback)
                rollback_update
                exit 0
                ;;
            *)
                echo "Uso: $0 [--skip-backup] [--force] [--rollback]"
                exit 1
                ;;
        esac
    done
    
    log "=== INICIANDO ACTUALIZACIÓN DEL SISTEMA ==="
    
    # Verificaciones pre-actualización
    if [ "$force_update" = false ]; then
        if ! pre_update_checks; then
            log "❌ Verificaciones pre-actualización fallaron"
            exit 1
        fi
    fi
    
    # Crear respaldo
    if [ "$skip_backup" = false ]; then
        create_pre_update_backup
    fi
    
    # Ejecutar actualización
    if update_system_dependencies && \
       update_application_code && \
       update_python_dependencies && \
       run_database_migrations && \
       update_static_files && \
       restart_services; then
        
        # Verificaciones post-actualización
        if post_update_checks; then
            log "✅ Actualización completada exitosamente"
            
            # Enviar notificación de éxito
            echo "Actualización de SIGeC-Balistica completada exitosamente.
Fecha: $(date)
Log: $UPDATE_LOG" | \
                mail -s "SIGeC-Balistica: Actualización exitosa" admin@company.com
        else
            log "❌ Verificaciones post-actualización fallaron"
            rollback_update
            exit 1
        fi
    else
        log "❌ Error durante la actualización"
        rollback_update
        exit 1
    fi
    
    log "=== ACTUALIZACIÓN FINALIZADA ==="
}

main "$@"
```

---

## Monitoreo y Alertas

### Sistema de Monitoreo Continuo

```python
#!/usr/bin/env python3
# continuous_monitor.py - Monitor continuo del sistema

import time
import psutil
import requests
import smtplib
import json
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración
CONFIG = {
    'check_interval': 60,  # segundos
    'thresholds': {
        'cpu_percent': 80,
        'memory_percent': 85,
        'disk_percent': 85,
        'response_time': 5.0  # segundos
    },
    'alerts': {
        'email': {
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'username': 'alerts@company.com',
            'password': 'password',
            'recipients': ['admin@company.com', 'devops@company.com']
        }
    },
    'services': [
        {'name': 'sigec-balistica', 'port': 8000, 'path': '/health'},
        {'name': 'postgresql', 'port': 5432},
        {'name': 'redis', 'port': 6379}
    ]
}

class SystemMonitor:
    def __init__(self):
        self.setup_logging()
        self.alert_history = {}
        self.last_alert_time = {}
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/sigec-balistica/monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_system_resources(self):
        """Verificar recursos del sistema"""
        metrics = {}
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu_percent'] = cpu_percent
        
        # Memoria
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        
        # Disco
        disk = psutil.disk_usage('/')
        metrics['disk_percent'] = disk.percent
        
        # Procesos
        sigec_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if 'sigec' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                sigec_processes.append(proc.info)
        
        metrics['processes'] = sigec_processes
        
        return metrics
    
    def check_services(self):
        """Verificar estado de servicios"""
        service_status = {}
        
        for service in CONFIG['services']:
            name = service['name']
            port = service['port']
            
            # Verificar puerto
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            service_status[name] = {
                'port_open': result == 0,
                'response_time': None
            }
            
            # Verificar endpoint HTTP si está definido
            if 'path' in service and result == 0:
                try:
                    start_time = time.time()
                    response = requests.get(f'http://localhost:{port}{service["path"]}', timeout=10)
                    response_time = time.time() - start_time
                    
                    service_status[name].update({
                        'http_status': response.status_code,
                        'response_time': response_time,
                        'healthy': response.status_code == 200
                    })
                except Exception as e:
                    service_status[name].update({
                        'http_status': None,
                        'response_time': None,
                        'healthy': False,
                        'error': str(e)
                    })
        
        return service_status
    
    def check_database_health(self):
        """Verificar salud de la base de datos"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host='localhost',
                database='sigec_balistica',
                user='sigec_user',
                password='password'  # Usar variable de entorno
            )
            
            cursor = conn.cursor()
            
            # Verificar conectividad básica
            cursor.execute('SELECT 1')
            
            # Verificar estadísticas básicas
            cursor.execute('SELECT COUNT(*) FROM samples')
            sample_count = cursor.fetchone()[0]
            
            # Verificar conexiones activas
            cursor.execute('SELECT COUNT(*) FROM pg_stat_activity WHERE state = %s', ('active',))
            active_connections = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                'connected': True,
                'sample_count': sample_count,
                'active_connections': active_connections
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
    
    def evaluate_alerts(self, metrics, services, database):
        """Evaluar si se deben enviar alertas"""
        alerts = []
        
        # Alertas de recursos
        if metrics['cpu_percent'] > CONFIG['thresholds']['cpu_percent']:
            alerts.append({
                'type': 'resource',
                'severity': 'warning',
                'message': f"CPU usage alto: {metrics['cpu_percent']:.1f}%"
            })
        
        if metrics['memory_percent'] > CONFIG['thresholds']['memory_percent']:
            alerts.append({
                'type': 'resource',
                'severity': 'warning',
                'message': f"Memoria usage alta: {metrics['memory_percent']:.1f}%"
            })
        
        if metrics['disk_percent'] > CONFIG['thresholds']['disk_percent']:
            alerts.append({
                'type': 'resource',
                'severity': 'critical',
                'message': f"Disco lleno: {metrics['disk_percent']:.1f}%"
            })
        
        # Alertas de servicios
        for service_name, status in services.items():
            if not status['port_open']:
                alerts.append({
                    'type': 'service',
                    'severity': 'critical',
                    'message': f"Servicio {service_name} no responde"
                })
            elif status.get('response_time') and status['response_time'] > CONFIG['thresholds']['response_time']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f"Servicio {service_name} respuesta lenta: {status['response_time']:.2f}s"
                })
        
        # Alertas de base de datos
        if not database['connected']:
            alerts.append({
                'type': 'database',
                'severity': 'critical',
                'message': f"Base de datos no disponible: {database.get('error', 'Unknown error')}"
            })
        
        return alerts
    
    def send_alert(self, alert):
        """Enviar alerta por email"""
        alert_key = f"{alert['type']}_{alert['message']}"
        current_time = time.time()
        
        # Evitar spam de alertas (mínimo 15 minutos entre alertas similares)
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < 900:  # 15 minutos
                return
        
        self.last_alert_time[alert_key] = current_time
        
        try:
            msg = MIMEMultipart()
            msg['From'] = CONFIG['alerts']['email']['username']
            msg['To'] = ', '.join(CONFIG['alerts']['email']['recipients'])
            msg['Subject'] = f"SIGeC-Balistica Alert: {alert['severity'].upper()}"
            
            body = f"""
Alerta del Sistema SIGeC-Balistica

Tipo: {alert['type']}
Severidad: {alert['severity']}
Mensaje: {alert['message']}
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Servidor: {psutil.os.uname().nodename}

Esta es una alerta automática del sistema de monitoreo.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(CONFIG['alerts']['email']['smtp_server'], CONFIG['alerts']['email']['smtp_port'])
            server.starttls()
            server.login(CONFIG['alerts']['email']['username'], CONFIG['alerts']['email']['password'])
            
            text = msg.as_string()
            server.sendmail(CONFIG['alerts']['email']['username'], CONFIG['alerts']['email']['recipients'], text)
            server.quit()
            
            self.logger.info(f"Alerta enviada: {alert['message']}")
            
        except Exception as e:
            self.logger.error(f"Error enviando alerta: {e}")
    
    def run(self):
        """Ejecutar monitoreo continuo"""
        self.logger.info("Iniciando monitor continuo del sistema")
        
        while True:
            try:
                # Recopilar métricas
                metrics = self.check_system_resources()
                services = self.check_services()
                database = self.check_database_health()
                
                # Log de estado
                self.logger.info(f"CPU: {metrics['cpu_percent']:.1f}%, "
                               f"RAM: {metrics['memory_percent']:.1f}%, "
                               f"Disk: {metrics['disk_percent']:.1f}%")
                
                # Evaluar alertas
                alerts = self.evaluate_alerts(metrics, services, database)
                
                # Enviar alertas
                for alert in alerts:
                    self.send_alert(alert)
                
                # Guardar métricas para análisis posterior
                self.save_metrics({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'services': services,
                    'database': database
                })
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de monitoreo: {e}")
            
            time.sleep(CONFIG['check_interval'])
    
    def save_metrics(self, data):
        """Guardar métricas para análisis posterior"""
        try:
            with open('/var/log/sigec-balistica/metrics.jsonl', 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            self.logger.error(f"Error guardando métricas: {e}")

if __name__ == '__main__':
    monitor = SystemMonitor()
    monitor.run()
```

---

## Optimización de Rendimiento

### Análisis de Rendimiento

```python
#!/usr/bin/env python3
# performance_analyzer.py - Analizador de rendimiento

import time
import psutil
import psycopg2
import json
import numpy as np
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics_history = []
        self.load_historical_data()
    
    def load_historical_data(self):
        """Cargar datos históricos de métricas"""
        try:
            with open('/var/log/sigec-balistica/metrics.jsonl', 'r') as f:
                for line in f:
                    self.metrics_history.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass
    
    def analyze_cpu_trends(self, days=7):
        """Analizar tendencias de CPU"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        if not recent_metrics:
            return None
        
        cpu_values = [m['metrics']['cpu_percent'] for m in recent_metrics]
        
        return {
            'average': np.mean(cpu_values),
            'max': np.max(cpu_values),
            'min': np.min(cpu_values),
            'std': np.std(cpu_values),
            'trend': self.calculate_trend(cpu_values),
            'peak_hours': self.find_peak_hours(recent_metrics, 'cpu_percent')
        }
    
    def analyze_memory_trends(self, days=7):
        """Analizar tendencias de memoria"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff_date
        ]
        
        if not recent_metrics:
            return None
        
        memory_values = [m['metrics']['memory_percent'] for m in recent_metrics]
        
        return {
            'average': np.mean(memory_values),
            'max': np.max(memory_values),
            'min': np.min(memory_values),
            'std': np.std(memory_values),
            'trend': self.calculate_trend(memory_values),
            'peak_hours': self.find_peak_hours(recent_metrics, 'memory_percent')
        }
    
    def analyze_database_performance(self):
        """Analizar rendimiento de base de datos"""
        try:
            conn = psycopg2.connect(
                host='localhost',
                database='sigec_balistica',
                user='sigec_user',
                password='password'
            )
            
            cursor = conn.cursor()
            
            # Consultas más lentas
            cursor.execute("""
                SELECT query, mean_time, calls, total_time
                FROM pg_stat_statements
                ORDER BY mean_time DESC
                LIMIT 10
            """)
            slow_queries = cursor.fetchall()
            
            # Tamaño de tablas
            cursor.execute("""
                SELECT schemaname, tablename, 
                       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            table_sizes = cursor.fetchall()
            
            # Índices no utilizados
            cursor.execute("""
                SELECT schemaname, tablename, indexname, idx_scan
                FROM pg_stat_user_indexes
                WHERE idx_scan = 0
            """)
            unused_indexes = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'slow_queries': slow_queries,
                'table_sizes': table_sizes,
                'unused_indexes': unused_indexes
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_trend(self, values):
        """Calcular tendencia de valores"""
        if len(values) < 2:
            return 'insufficient_data'
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def find_peak_hours(self, metrics, metric_name):
        """Encontrar horas pico de uso"""
        hourly_averages = {}
        
        for metric in metrics:
            timestamp = datetime.fromisoformat(metric['timestamp'])
            hour = timestamp.hour
            value = metric['metrics'][metric_name]
            
            if hour not in hourly_averages:
                hourly_averages[hour] = []
            hourly_averages[hour].append(value)
        
        # Calcular promedio por hora
        for hour in hourly_averages:
            hourly_averages[hour] = np.mean(hourly_averages[hour])
        
        # Encontrar las 3 horas con mayor uso
        sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1], reverse=True)
        return sorted_hours[:3]
    
    def generate_recommendations(self):
        """Generar recomendaciones de optimización"""
        recommendations = []
        
        # Análisis de CPU
        cpu_analysis = self.analyze_cpu_trends()
        if cpu_analysis and cpu_analysis['average'] > 70:
            recommendations.append({
                'category': 'CPU',
                'priority': 'high',
                'recommendation': 'Considerar aumentar recursos de CPU o optimizar algoritmos',
                'details': f"Uso promedio de CPU: {cpu_analysis['average']:.1f}%"
            })
        
        # Análisis de memoria
        memory_analysis = self.analyze_memory_trends()
        if memory_analysis and memory_analysis['average'] > 80:
            recommendations.append({
                'category': 'Memory',
                'priority': 'high',
                'recommendation': 'Aumentar memoria RAM o implementar mejor gestión de memoria',
                'details': f"Uso promedio de memoria: {memory_analysis['average']:.1f}%"
            })
        
        # Análisis de base de datos
        db_analysis = self.analyze_database_performance()
        if 'slow_queries' in db_analysis and db_analysis['slow_queries']:
            recommendations.append({
                'category': 'Database',
                'priority': 'medium',
                'recommendation': 'Optimizar consultas lentas de base de datos',
                'details': f"Encontradas {len(db_analysis['slow_queries'])} consultas lentas"
            })
        
        if 'unused_indexes' in db_analysis and db_analysis['unused_indexes']:
            recommendations.append({
                'category': 'Database',
                'priority': 'low',
                'recommendation': 'Eliminar índices no utilizados para mejorar rendimiento de escritura',
                'details': f"Encontrados {len(db_analysis['unused_indexes'])} índices no utilizados"
            })
        
        return recommendations
    
    def generate_report(self):
        """Generar reporte completo de rendimiento"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cpu_analysis': self.analyze_cpu_trends(),
            'memory_analysis': self.analyze_memory_trends(),
            'database_analysis': self.analyze_database_performance(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def save_report(self, report):
        """Guardar reporte de rendimiento"""
        filename = f"/var/log/sigec-balistica/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename

if __name__ == '__main__':
    analyzer = PerformanceAnalyzer()
    report = analyzer.generate_report()
    filename = analyzer.save_report(report)
    
    print(f"Reporte de rendimiento generado: {filename}")
    
    # Mostrar resumen
    print("\n=== RESUMEN DE RENDIMIENTO ===")
    
    if report['cpu_analysis']:
        print(f"CPU promedio: {report['cpu_analysis']['average']:.1f}%")
        print(f"CPU tendencia: {report['cpu_analysis']['trend']}")
    
    if report['memory_analysis']:
        print(f"Memoria promedio: {report['memory_analysis']['average']:.1f}%")
        print(f"Memoria tendencia: {report['memory_analysis']['trend']}")
    
    print(f"\nRecomendaciones: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"- [{rec['priority'].upper()}] {rec['recommendation']}")
```

---

## Gestión de Logs

### Rotación y Limpieza de Logs

```bash
#!/bin/bash
# log_management.sh - Gestión de logs del sistema

LOG_BASE_DIR="/var/log/sigec-balistica"
RETENTION_DAYS=30
ARCHIVE_DAYS=90
MAX_LOG_SIZE="100M"

# Configurar logrotate
setup_logrotate() {
    cat > /etc/logrotate.d/sigec-balistica << EOF
$LOG_BASE_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 sigec sigec
    postrotate
        systemctl reload sigec-balistica > /dev/null 2>&1 || true
    endscript
}

$LOG_BASE_DIR/access.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 sigec sigec
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
    endscript
}
EOF

    echo "✅ Configuración de logrotate actualizada"
}

# Limpiar logs antiguos
cleanup_old_logs() {
    echo "Limpiando logs antiguos..."
    
    # Eliminar logs más antiguos que RETENTION_DAYS
    find "$LOG_BASE_DIR" -name "*.log.*" -mtime +$RETENTION_DAYS -delete
    
    # Archivar logs entre RETENTION_DAYS y ARCHIVE_DAYS
    find "$LOG_BASE_DIR" -name "*.log" -mtime +$RETENTION_DAYS -mtime -$ARCHIVE_DAYS -exec gzip {} \;
    
    # Eliminar archivos muy antiguos
    find "$LOG_BASE_DIR" -name "*.gz" -mtime +$ARCHIVE_DAYS -delete
    
    echo "✅ Limpieza de logs completada"
}

# Analizar uso de espacio en logs
analyze_log_usage() {
    echo "=== ANÁLISIS DE USO DE LOGS ==="
    
    echo "Tamaño total de logs:"
    du -sh "$LOG_BASE_DIR"
    
    echo -e "\nArchivos más grandes:"
    find "$LOG_BASE_DIR" -type f -exec du -h {} \; | sort -hr | head -10
    
    echo -e "\nDistribución por tipo:"
    find "$LOG_BASE_DIR" -name "*.log" -exec du -ch {} \; | tail -1 | awk '{print "Logs activos: " $1}'
    find "$LOG_BASE_DIR" -name "*.gz" -exec du -ch {} \; | tail -1 | awk '{print "Logs comprimidos: " $1}'
}

# Extraer métricas de logs
extract_log_metrics() {
    local log_file="$1"
    local output_file="$2"
    
    echo "Extrayendo métricas de: $(basename $log_file)"
    
    # Análisis de errores por hora
    echo "=== ERRORES POR HORA ===" > "$output_file"
    grep "ERROR" "$log_file" | cut -d' ' -f1-2 | cut -d':' -f1-2 | sort | uniq -c >> "$output_file"
    
    # Top errores
    echo -e "\n=== TOP ERRORES ===" >> "$output_file"
    grep "ERROR" "$log_file" | cut -d'-' -f4- | sort | uniq -c | sort -nr | head -10 >> "$output_file"
    
    # Análisis de rendimiento
    echo -e "\n=== ANÁLISIS DE RENDIMIENTO ===" >> "$output_file"
    grep "processing_time" "$log_file" | \
        sed 's/.*processing_time:\([0-9.]*\).*/\1/' | \
        awk '{sum+=$1; count++; if($1>max) max=$1; if(min=="" || $1<min) min=$1} 
             END {if(count>0) print "Promedio: " sum/count "s\nMáximo: " max "s\nMínimo: " min "s\nTotal: " count " operaciones"}' >> "$output_file"
    
    echo "✅ Métricas extraídas a: $output_file"
}

# Monitorear logs en tiempo real
monitor_logs() {
    echo "Monitoreando logs en tiempo real (Ctrl+C para salir)..."
    
    # Mostrar errores en tiempo real
    tail -f "$LOG_BASE_DIR"/*.log | grep --line-buffered "ERROR\|CRITICAL" | \
        while read line; do
            echo "[$(date '+%H:%M:%S')] $line"
        done
}

# Función principal
main() {
    case "$1" in
        "setup")
            setup_logrotate
            ;;
        "cleanup")
            cleanup_old_logs
            ;;
        "analyze")
            analyze_log_usage
            ;;
        "extract")
            if [ -z "$2" ] || [ -z "$3" ]; then
                echo "Uso: $0 extract <archivo_log> <archivo_salida>"
                exit 1
            fi
            extract_log_metrics "$2" "$3"
            ;;
        "monitor")
            monitor_logs
            ;;
        *)
            echo "Uso: $0 {setup|cleanup|analyze|extract|monitor}"
            echo
            echo "Comandos:"
            echo "  setup   - Configurar rotación automática de logs"
            echo "  cleanup - Limpiar logs antiguos"
            echo "  analyze - Analizar uso de espacio en logs"
            echo "  extract - Extraer métricas de un archivo de log"
            echo "  monitor - Monitorear logs en tiempo real"
            exit 1
            ;;
    esac
}

main "$@"
```

---

## Seguridad y Auditoría

### Auditoría de Seguridad

```bash
#!/bin/bash
# security_audit.sh - Auditoría de seguridad del sistema

AUDIT_LOG="/var/log/sigec-balistica/security_audit_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$AUDIT_LOG"
}

# Verificar permisos de archivos
check_file_permissions() {
    log "=== VERIFICACIÓN DE PERMISOS DE ARCHIVOS ==="
    
    # Archivos de configuración
    local config_files=(
        "/etc/sigec-balistica/config.yaml"
        "/etc/sigec-balistica/database.conf"
        "/etc/sigec-balistica/secrets.env"
    )
    
    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            local perms=$(stat -c "%a" "$file")
            local owner=$(stat -c "%U:%G" "$file")
            
            if [ "$perms" -gt 640 ]; then
                log "⚠️  Permisos demasiado amplios en $file: $perms ($owner)"
            else
                log "✅ Permisos correctos en $file: $perms ($owner)"
            fi
        else
            log "ℹ️  Archivo no encontrado: $file"
        fi
    done
    
    # Directorios sensibles
    local sensitive_dirs=(
        "/var/lib/sigec-balistica"
        "/var/log/sigec-balistica"
        "/etc/sigec-balistica"
    )
    
    for dir in "${sensitive_dirs[@]}"; do
        if [ -d "$dir" ]; then
            local perms=$(stat -c "%a" "$dir")
            local owner=$(stat -c "%U:%G" "$dir")
            
            if [ "$perms" -gt 750 ]; then
                log "⚠️  Permisos demasiado amplios en directorio $dir: $perms ($owner)"
            else
                log "✅ Permisos correctos en directorio $dir: $perms ($owner)"
            fi
        fi
    done
}

# Verificar usuarios y grupos
check_users_and_groups() {
    log "=== VERIFICACIÓN DE USUARIOS Y GRUPOS ==="
    
    # Verificar usuario sigec
    if id "sigec" &>/dev/null; then
        local sigec_groups=$(groups sigec | cut -d: -f2)
        log "✅ Usuario sigec existe. Grupos: $sigec_groups"
        
        # Verificar que no tenga privilegios sudo
        if sudo -l -U sigec 2>/dev/null | grep -q "may run"; then
            log "⚠️  Usuario sigec tiene privilegios sudo"
        else
            log "✅ Usuario sigec no tiene privilegios sudo"
        fi
    else
        log "❌ Usuario sigec no existe"
    fi
    
    # Verificar usuarios con acceso sudo
    log "Usuarios con acceso sudo:"
    getent group sudo | cut -d: -f4 | tr ',' '\n' | while read user; do
        if [ -n "$user" ]; then
            log "  - $user"
        fi
    done
}

# Verificar configuración de red
check_network_security() {
    log "=== VERIFICACIÓN DE SEGURIDAD DE RED ==="
    
    # Puertos abiertos
    log "Puertos abiertos:"
    netstat -tlnp | grep LISTEN | while read line; do
        local port=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
        local process=$(echo "$line" | awk '{print $7}')
        log "  Puerto $port: $process"
    done
    
    # Verificar firewall
    if command -v ufw &> /dev/null; then
        local ufw_status=$(ufw status | head -1)
        log "Estado UFW: $ufw_status"
        
        if echo "$ufw_status" | grep -q "inactive"; then
            log "⚠️  Firewall UFW está inactivo"
        else
            log "✅ Firewall UFW está activo"
        fi
    else
        log "ℹ️  UFW no está instalado"
    fi
    
    # Verificar iptables
    local iptables_rules=$(iptables -L | wc -l)
    if [ "$iptables_rules" -gt 8 ]; then
        log "✅ Reglas de iptables configuradas ($iptables_rules reglas)"
    else
        log "⚠️  Pocas reglas de iptables configuradas ($iptables_rules reglas)"
    fi
}

# Verificar certificados SSL
check_ssl_certificates() {
    log "=== VERIFICACIÓN DE CERTIFICADOS SSL ==="
    
    local cert_files=(
        "/etc/ssl/certs/sigec.crt"
        "/etc/nginx/ssl/sigec.crt"
        "/etc/letsencrypt/live/sigec.company.com/fullchain.pem"
    )
    
    for cert_file in "${cert_files[@]}"; do
        if [ -f "$cert_file" ]; then
            local expiry_date=$(openssl x509 -in "$cert_file" -noout -enddate 2>/dev/null | cut -d= -f2)
            
            if [ -n "$expiry_date" ]; then
                local expiry_timestamp=$(date -d "$expiry_date" +%s)
                local current_timestamp=$(date +%s)
                local days_left=$(( (expiry_timestamp - current_timestamp) / 86400 ))
                
                if [ "$days_left" -lt 30 ]; then
                    log "⚠️  Certificado $cert_file expira en $days_left días"
                elif [ "$days_left" -lt 0 ]; then
                    log "❌ Certificado $cert_file ha expirado"
                else
                    log "✅ Certificado $cert_file válido por $days_left días"
                fi
            else
                log "❌ Error leyendo certificado $cert_file"
            fi
        fi
    done
}

# Verificar logs de seguridad
check_security_logs() {
    log "=== VERIFICACIÓN DE LOGS DE SEGURIDAD ==="
    
    # Intentos de login fallidos
    local failed_logins=$(grep "Failed password" /var/log/auth.log 2>/dev/null | wc -l)
    if [ "$failed_logins" -gt 10 ]; then
        log "⚠️  Múltiples intentos de login fallidos: $failed_logins"
        
        # Mostrar IPs más frecuentes
        log "IPs con más intentos fallidos:"
        grep "Failed password" /var/log/auth.log 2>/dev/null | \
            grep -oE "from [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | \
            cut -d' ' -f2 | sort | uniq -c | sort -nr | head -5 | \
            while read count ip; do
                log "  $ip: $count intentos"
            done
    else
        log "✅ Intentos de login fallidos normales: $failed_logins"
    fi
    
    # Verificar accesos sudo
    local sudo_accesses=$(grep "sudo:" /var/log/auth.log 2>/dev/null | wc -l)
    log "Accesos sudo recientes: $sudo_accesses"
    
    # Verificar cambios en archivos críticos
    if command -v aide &> /dev/null; then
        log "Ejecutando verificación AIDE..."
        aide --check 2>/dev/null | tail -10 | while read line; do
            log "AIDE: $line"
        done
    else
        log "ℹ️  AIDE no está instalado para verificación de integridad"
    fi
}

# Verificar configuración de base de datos
check_database_security() {
    log "=== VERIFICACIÓN DE SEGURIDAD DE BASE DE DATOS ==="
    
    # Verificar configuración de PostgreSQL
    local pg_config="/etc/postgresql/*/main/postgresql.conf"
    if ls $pg_config 1> /dev/null 2>&1; then
        # Verificar listen_addresses
        local listen_addresses=$(grep "^listen_addresses" $pg_config | head -1)
        if echo "$listen_addresses" | grep -q "'\*'"; then
            log "⚠️  PostgreSQL escucha en todas las interfaces"
        else
            log "✅ PostgreSQL configurado para escuchar solo en interfaces específicas"
        fi
        
        # Verificar SSL
        local ssl_setting=$(grep "^ssl" $pg_config | head -1)
        if echo "$ssl_setting" | grep -q "on"; then
            log "✅ SSL habilitado en PostgreSQL"
        else
            log "⚠️  SSL no habilitado en PostgreSQL"
        fi
    fi
    
    # Verificar usuarios de base de datos
    if command -v psql &> /dev/null; then
        log "Usuarios de base de datos:"
        psql -d sigec_balistica -c "\du" 2>/dev/null | grep -E "^\s*\w+" | while read line; do
            log "  $line"
        done
    fi
}

# Generar reporte de seguridad
generate_security_report() {
    log "=== REPORTE DE SEGURIDAD GENERADO ==="
    log "Archivo de auditoría: $AUDIT_LOG"
    
    # Contar problemas encontrados
    local warnings=$(grep -c "⚠️" "$AUDIT_LOG")
    local errors=$(grep -c "❌" "$AUDIT_LOG")
    local successes=$(grep -c "✅" "$AUDIT_LOG")
    
    log "Resumen:"
    log "  ✅ Verificaciones exitosas: $successes"
    log "  ⚠️  Advertencias: $warnings"
    log "  ❌ Errores críticos: $errors"
    
    # Calcular puntuación de seguridad
    local total_checks=$((successes + warnings + errors))
    if [ "$total_checks" -gt 0 ]; then
        local security_score=$(( (successes * 100) / total_checks ))
        log "Puntuación de seguridad: $security_score/100"
        
        if [ "$security_score" -ge 80 ]; then
            log "✅ Nivel de seguridad: BUENO"
        elif [ "$security_score" -ge 60 ]; then
            log "⚠️  Nivel de seguridad: REGULAR"
        else
            log "❌ Nivel de seguridad: CRÍTICO"
        fi
    fi
}

# Función principal
main() {
    log "=== INICIANDO AUDITORÍA DE SEGURIDAD ==="
    
    check_file_permissions
    check_users_and_groups
    check_network_security
    check_ssl_certificates
    check_security_logs
    check_database_security
    generate_security_report
    
    log "=== AUDITORÍA DE SEGURIDAD COMPLETADA ==="
    
    echo "Reporte de auditoría guardado en: $AUDIT_LOG"
}

main "$@"
```

---

## Procedimientos de Emergencia

### Plan de Respuesta a Incidentes

```bash
#!/bin/bash
# emergency_response.sh - Plan de respuesta a emergencias

EMERGENCY_LOG="/var/log/sigec-balistica/emergency_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="/backup/sigec-balistica/emergency"
NOTIFICATION_LIST="admin@company.com,devops@company.com,manager@company.com"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$EMERGENCY_LOG"
}

send_emergency_notification() {
    local incident_type="$1"
    local severity="$2"
    local description="$3"
    
    local subject="🚨 EMERGENCIA SIGeC-Balistica: $incident_type ($severity)"
    local body="ALERTA DE EMERGENCIA

Tipo de Incidente: $incident_type
Severidad: $severity
Descripción: $description
Servidor: $(hostname)
Fecha/Hora: $(date)
Log de Emergencia: $EMERGENCY_LOG

Este es un mensaje automático del sistema de respuesta a emergencias.
Se requiere atención inmediata.

Procedimientos de respuesta activados automáticamente.
"

    echo "$body" | mail -s "$subject" "$NOTIFICATION_LIST"
    log "📧 Notificación de emergencia enviada: $incident_type"
}

# Respuesta a caída del sistema
system_down_response() {
    log "🚨 RESPUESTA A CAÍDA DEL SISTEMA"
    
    send_emergency_notification "SYSTEM_DOWN" "CRITICAL" "Sistema principal no responde"
    
    # Crear respaldo de emergencia
    mkdir -p "$BACKUP_DIR"
    
    # Respaldo rápido de base de datos
    if pg_isready -h localhost -p 5432; then
        log "Creando respaldo de emergencia de BD..."
        pg_dump sigec_balistica > "$BACKUP_DIR/emergency_db_$(date +%Y%m%d_%H%M%S).sql"
        log "✅ Respaldo de BD completado"
    else
        log "❌ Base de datos no disponible para respaldo"
    fi
    
    # Intentar reinicio de servicios
    log "Intentando reinicio de servicios..."
    
    for service in postgresql redis sigec-balistica nginx; do
        log "Reiniciando $service..."
        systemctl restart "$service"
        sleep 5
        
        if systemctl is-active --quiet "$service"; then
            log "✅ $service reiniciado exitosamente"
        else
            log "❌ Fallo al reiniciar $service"
        fi
    done
    
    # Verificar recuperación
    sleep 30
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log "✅ Sistema recuperado automáticamente"
        send_emergency_notification "SYSTEM_RECOVERED" "INFO" "Sistema recuperado tras reinicio automático"
    else
        log "❌ Sistema no se recuperó automáticamente"
        escalate_incident "SYSTEM_DOWN" "Sistema no se recuperó tras reinicio automático"
    fi
}

# Respuesta a alta carga del sistema
high_load_response() {
    local cpu_usage="$1"
    local memory_usage="$2"
    
    log "🚨 RESPUESTA A ALTA CARGA DEL SISTEMA"
    log "CPU: ${cpu_usage}%, Memoria: ${memory_usage}%"
    
    send_emergency_notification "HIGH_LOAD" "WARNING" "CPU: ${cpu_usage}%, Memoria: ${memory_usage}%"
    
    # Identificar procesos que consumen más recursos
    log "Procesos con mayor consumo de CPU:"
    ps aux --sort=-%cpu | head -10 | while read line; do
        log "  $line"
    done
    
    log "Procesos con mayor consumo de memoria:"
    ps aux --sort=-%mem | head -10 | while read line; do
        log "  $line"
    done
    
    # Acciones de mitigación
    log "Aplicando medidas de mitigación..."
    
    # Limpiar caché del sistema
    sync && echo 3 > /proc/sys/vm/drop_caches
    log "✅ Caché del sistema limpiado"
    
    # Reiniciar servicios no críticos
    systemctl restart redis
    log "✅ Redis reiniciado"
    
    # Verificar mejora
    sleep 60
    local new_cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    local new_memory=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    
    log "Nuevos valores - CPU: ${new_cpu}%, Memoria: ${new_memory}%"
    
    if (( $(echo "$new_cpu < 70" | bc -l) )) && (( $(echo "$new_memory < 80" | bc -l) )); then
        log "✅ Carga del sistema normalizada"
        send_emergency_notification "LOAD_NORMALIZED" "INFO" "Carga del sistema normalizada tras medidas de mitigación"
    else
        log "⚠️  Carga del sistema sigue alta"
        escalate_incident "HIGH_LOAD" "Carga del sistema no se normalizó tras medidas de mitigación"
    fi
}

# Respuesta a fallo de base de datos
database_failure_response() {
    log "🚨 RESPUESTA A FALLO DE BASE DE DATOS"
    
    send_emergency_notification "DATABASE_FAILURE" "CRITICAL" "Base de datos no disponible"
    
    # Verificar estado de PostgreSQL
    if ! systemctl is-active --quiet postgresql; then
        log "PostgreSQL no está activo, intentando reinicio..."
        systemctl restart postgresql
        sleep 10
        
        if systemctl is-active --quiet postgresql; then
            log "✅ PostgreSQL reiniciado exitosamente"
        else
            log "❌ Fallo al reiniciar PostgreSQL"
            escalate_incident "DATABASE_FAILURE" "No se pudo reiniciar PostgreSQL"
            return 1
        fi
    fi
    
    # Verificar conectividad
    if pg_isready -h localhost -p 5432; then
        log "✅ PostgreSQL responde"
        
        # Verificar integridad de la base de datos
        log "Verificando integridad de la base de datos..."
        if psql -d sigec_balistica -c "SELECT 1;" > /dev/null 2>&1; then
            log "✅ Base de datos accesible"
            
            # Verificar tablas principales
            local table_count=$(psql -d sigec_balistica -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')
            
            if [ "$table_count" -gt 0 ]; then
                log "✅ Tablas de base de datos presentes ($table_count tablas)"
                send_emergency_notification "DATABASE_RECOVERED" "INFO" "Base de datos recuperada automáticamente"
            else
                log "❌ No se encontraron tablas en la base de datos"
                escalate_incident "DATABASE_CORRUPTION" "Base de datos sin tablas"
            fi
        else
            log "❌ No se puede acceder a la base de datos"
            escalate_incident "DATABASE_ACCESS" "No se puede acceder a la base de datos"
        fi
    else
        log "❌ PostgreSQL no responde"
        escalate_incident "DATABASE_FAILURE" "PostgreSQL no responde tras reinicio"
    fi
}

# Respuesta a fallo de seguridad
security_breach_response() {
    local breach_type="$1"
    local details="$2"
    
    log "🚨 RESPUESTA A FALLO DE SEGURIDAD"
    log "Tipo: $breach_type"
    log "Detalles: $details"
    
    send_emergency_notification "SECURITY_BREACH" "CRITICAL" "$breach_type: $details"
    
    # Medidas inmediatas de seguridad
    log "Aplicando medidas de seguridad inmediatas..."
    
    # Bloquear IPs sospechosas si se detectan
    if [[ "$details" == *"IP:"* ]]; then
        local suspicious_ip=$(echo "$details" | grep -oE "IP: [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | cut -d' ' -f2)
        if [ -n "$suspicious_ip" ]; then
            log "Bloqueando IP sospechosa: $suspicious_ip"
            iptables -A INPUT -s "$suspicious_ip" -j DROP
            log "✅ IP $suspicious_ip bloqueada"
        fi
    fi
    
    # Cambiar contraseñas de emergencia
    log "Generando contraseñas de emergencia..."
    local new_password=$(openssl rand -base64 32)
    echo "sigec_user:$new_password" | chpasswd
    log "✅ Contraseña de usuario sigec_user cambiada"
    
    # Crear respaldo de seguridad
    log "Creando respaldo de seguridad..."
    mkdir -p "$BACKUP_DIR/security"
    
    # Respaldo de logs de seguridad
    cp /var/log/auth.log "$BACKUP_DIR/security/auth_$(date +%Y%m%d_%H%M%S).log"
    cp /var/log/sigec-balistica/*.log "$BACKUP_DIR/security/" 2>/dev/null
    
    # Respaldo de configuración
    tar -czf "$BACKUP_DIR/security/config_$(date +%Y%m%d_%H%M%S).tar.gz" \
        /etc/sigec-balistica \
        /etc/nginx/sites-available/sigec* \
        /etc/ssl/certs/sigec* 2>/dev/null
    
    log "✅ Respaldos de seguridad completados"
    
    # Activar modo de mantenimiento
    log "Activando modo de mantenimiento..."
    touch /var/lib/sigec-balistica/maintenance_mode
    systemctl reload nginx
    log "✅ Modo de mantenimiento activado"
    
    escalate_incident "SECURITY_BREACH" "$breach_type: $details - Medidas de seguridad aplicadas"
}

# Escalamiento de incidentes
escalate_incident() {
    local incident_type="$1"
    local description="$2"
    
    log "📈 ESCALAMIENTO DE INCIDENTE: $incident_type"
    
    # Notificar a nivel gerencial
    local escalation_subject="🔴 ESCALAMIENTO CRÍTICO SIGeC-Balistica: $incident_type"
    local escalation_body="ESCALAMIENTO DE INCIDENTE CRÍTICO

Tipo de Incidente: $incident_type
Descripción: $description
Servidor: $(hostname)
Fecha/Hora: $(date)
Log de Emergencia: $EMERGENCY_LOG

ACCIONES REQUERIDAS:
1. Revisión inmediata por parte del equipo técnico
2. Evaluación de impacto en el negocio
3. Comunicación con stakeholders
4. Plan de recuperación manual si es necesario

Estado del Sistema:
- Servicios: $(systemctl is-active sigec-balistica postgresql redis nginx | tr '\n' ' ')
- Carga CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
- Memoria: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')
- Disco: $(df -h / | awk 'NR==2{print $5}')

Este incidente requiere atención inmediata de nivel gerencial.
"

    echo "$escalation_body" | mail -s "$escalation_subject" "manager@company.com,cto@company.com"
    
    # Crear ticket de emergencia (integración con sistema de tickets)
    create_emergency_ticket "$incident_type" "$description"
    
    log "📧 Incidente escalado a nivel gerencial"
}

# Crear ticket de emergencia
create_emergency_ticket() {
    local incident_type="$1"
    local description="$2"
    
    # Ejemplo de integración con sistema de tickets (adaptar según el sistema usado)
    local ticket_data="{
        \"title\": \"EMERGENCIA: $incident_type\",
        \"description\": \"$description\",
        \"priority\": \"critical\",
        \"category\": \"system_emergency\",
        \"assigned_to\": \"devops_team\",
        \"created_by\": \"emergency_system\",
        \"tags\": [\"emergency\", \"sigec-balistica\", \"critical\"]
    }"
    
    # Enviar a sistema de tickets (ejemplo con curl)
    # curl -X POST -H "Content-Type: application/json" \
    #      -d "$ticket_data" \
    #      "https://tickets.company.com/api/v1/tickets"
    
    log "🎫 Ticket de emergencia creado: $incident_type"
}

# Función principal
main() {
    local incident_type="$1"
    local param1="$2"
    local param2="$3"
    
    log "=== ACTIVACIÓN DE RESPUESTA DE EMERGENCIA ==="
    log "Tipo de incidente: $incident_type"
    
    case "$incident_type" in
        "system_down")
            system_down_response
            ;;
        "high_load")
            high_load_response "$param1" "$param2"
            ;;
        "database_failure")
            database_failure_response
            ;;
        "security_breach")
            security_breach_response "$param1" "$param2"
            ;;
        *)
            echo "Uso: $0 {system_down|high_load|database_failure|security_breach} [parámetros]"
            echo
            echo "Tipos de incidente:"
            echo "  system_down                    - Sistema principal no responde"
            echo "  high_load <cpu%> <memory%>     - Alta carga del sistema"
            echo "  database_failure               - Fallo de base de datos"
            echo "  security_breach <type> <details> - Fallo de seguridad"
            echo
            echo "Ejemplos:"
            echo "  $0 system_down"
            echo "  $0 high_load 85 90"
            echo "  $0 database_failure"
            echo "  $0 security_breach 'brute_force' 'IP: 192.168.1.100'"
            exit 1
            ;;
    esac
    
    log "=== RESPUESTA DE EMERGENCIA COMPLETADA ==="
}

main "$@"
```

---

## Conclusión

Este documento de procedimientos de mantenimiento proporciona una guía completa para mantener el sistema SIGeC-Balistica funcionando de manera óptima y segura. Los procedimientos incluyen:

### Puntos Clave:

1. **Mantenimiento Preventivo**: Cronogramas y listas de verificación automatizadas
2. **Respaldos y Recuperación**: Estrategias completas con scripts automatizados
3. **Actualizaciones**: Procedimientos seguros con rollback automático
4. **Monitoreo**: Sistema continuo con alertas inteligentes
5. **Optimización**: Análisis de rendimiento y recomendaciones
6. **Gestión de Logs**: Rotación, análisis y monitoreo en tiempo real
7. **Seguridad**: Auditorías regulares y verificaciones de cumplimiento
8. **Emergencias**: Planes de respuesta automática y escalamiento

### Recomendaciones de Implementación:

1. **Automatización**: Configurar todos los scripts en cron jobs
2. **Monitoreo**: Implementar el sistema de monitoreo continuo
3. **Documentación**: Mantener este documento actualizado
4. **Capacitación**: Entrenar al equipo en estos procedimientos
5. **Pruebas**: Realizar simulacros regulares de emergencia

### Próximos Pasos:

1. Configurar la automatización de todos los scripts
2. Implementar el sistema de alertas
3. Realizar pruebas de los procedimientos de recuperación
4. Establecer métricas de rendimiento baseline
5. Programar auditorías de seguridad regulares

Este sistema de mantenimiento asegura la disponibilidad, seguridad y rendimiento óptimo del sistema SIGeC-Balistica en producción.