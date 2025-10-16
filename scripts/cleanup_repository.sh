#!/bin/bash

# =============================================================================
# SCRIPT DE LIMPIEZA AUTOMÁTICA DEL REPOSITORIO SIGeC-Balistica
# =============================================================================
# Fecha: Diciembre 2024
# Propósito: Eliminar archivos obsoletos, cache y directorios vacíos
# Basado en: Análisis completo del repositorio
# =============================================================================

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Banner
echo "============================================================================="
echo "🧹 LIMPIEZA AUTOMÁTICA DEL REPOSITORIO SIGeC-Balistica"
echo "============================================================================="
echo ""

# Verificar que estamos en el directorio correcto
if [[ ! -f "main.py" ]] || [[ ! -d "gui" ]] || [[ ! -d "matching" ]]; then
    error "Error: No se detectó el directorio raíz de SIGeC-Balistica"
    error "Ejecute este script desde el directorio raíz del proyecto"
    exit 1
fi

log "Iniciando limpieza del repositorio SIGeC-Balistica..."

# =============================================================================
# 1. ELIMINAR ARCHIVOS CACHE DE PYTHON
# =============================================================================
log "🗑️  Eliminando archivos cache de Python..."

# Contar archivos antes de eliminar
cache_count=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
pyc_count=$(find . -name "*.pyc" 2>/dev/null | wc -l)
pyo_count=$(find . -name "*.pyo" 2>/dev/null | wc -l)

if [[ $cache_count -gt 0 ]]; then
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    success "Eliminados $cache_count directorios __pycache__"
else
    log "No se encontraron directorios __pycache__"
fi

if [[ $pyc_count -gt 0 ]]; then
    find . -name "*.pyc" -delete 2>/dev/null || true
    success "Eliminados $pyc_count archivos .pyc"
else
    log "No se encontraron archivos .pyc"
fi

if [[ $pyo_count -gt 0 ]]; then
    find . -name "*.pyo" -delete 2>/dev/null || true
    success "Eliminados $pyo_count archivos .pyo"
else
    log "No se encontraron archivos .pyo"
fi

# =============================================================================
# 2. ELIMINAR CACHE DE PYTEST
# =============================================================================
log "🧪 Eliminando cache de pytest..."

pytest_count=$(find . -name ".pytest_cache" -type d 2>/dev/null | wc -l)
if [[ $pytest_count -gt 0 ]]; then
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    success "Eliminados $pytest_count directorios .pytest_cache"
else
    log "No se encontraron directorios .pytest_cache"
fi

# =============================================================================
# 3. LIMPIAR LOGS ANTIGUOS
# =============================================================================
log "📝 Limpiando logs antiguos (>30 días)..."

if [[ -d "logs" ]]; then
    old_logs=$(find logs/ -name "*.log" -mtime +30 2>/dev/null | wc -l)
    if [[ $old_logs -gt 0 ]]; then
        find logs/ -name "*.log" -mtime +30 -delete 2>/dev/null || true
        success "Eliminados $old_logs archivos de log antiguos"
    else
        log "No se encontraron logs antiguos para eliminar"
    fi
else
    log "Directorio logs/ no existe"
fi

# =============================================================================
# 4. LIMPIAR RESULTADOS TEMPORALES
# =============================================================================
log "🔬 Limpiando resultados temporales..."

# Simple test results temporales
if [[ -d "simple_test_results" ]]; then
    temp_simple=$(find simple_test_results/ -name "temp_*" 2>/dev/null | wc -l)
    if [[ $temp_simple -gt 0 ]]; then
        rm -rf simple_test_results/temp_* 2>/dev/null || true
        success "Eliminados $temp_simple archivos temporales de simple_test_results"
    fi
fi

# Advanced test results temporales
if [[ -d "advanced_test_results" ]]; then
    temp_advanced=$(find advanced_test_results/ -name "temp_*" 2>/dev/null | wc -l)
    if [[ $temp_advanced -gt 0 ]]; then
        rm -rf advanced_test_results/temp_* 2>/dev/null || true
        success "Eliminados $temp_advanced archivos temporales de advanced_test_results"
    fi
fi

# =============================================================================
# 5. ELIMINAR DIRECTORIOS VACÍOS
# =============================================================================
log "📁 Eliminando directorios vacíos..."

# Función para eliminar directorios vacíos de forma segura
remove_empty_dirs() {
    local removed=0
    # Buscar directorios vacíos (excluyendo .git y venv)
    while IFS= read -r -d '' dir; do
        # Verificar que no sea un directorio crítico
        if [[ "$dir" != *".git"* ]] && [[ "$dir" != *"venv"* ]] && [[ "$dir" != *"node_modules"* ]]; then
            if rmdir "$dir" 2>/dev/null; then
                ((removed++))
                log "Eliminado directorio vacío: $dir"
            fi
        fi
    done < <(find . -type d -empty -print0 2>/dev/null)
    
    if [[ $removed -gt 0 ]]; then
        success "Eliminados $removed directorios vacíos"
    else
        log "No se encontraron directorios vacíos para eliminar"
    fi
}

remove_empty_dirs

# =============================================================================
# 6. LIMPIAR ARCHIVOS DE BACKUP Y TEMPORALES
# =============================================================================
log "💾 Limpiando archivos de backup y temporales..."

# Archivos de backup comunes
backup_patterns=("*.bak" "*.backup" "*~" "*.tmp" "*.temp" ".DS_Store" "Thumbs.db")
total_backup=0

for pattern in "${backup_patterns[@]}"; do
    count=$(find . -name "$pattern" -type f 2>/dev/null | wc -l)
    if [[ $count -gt 0 ]]; then
        find . -name "$pattern" -type f -delete 2>/dev/null || true
        total_backup=$((total_backup + count))
    fi
done

if [[ $total_backup -gt 0 ]]; then
    success "Eliminados $total_backup archivos de backup y temporales"
else
    log "No se encontraron archivos de backup para eliminar"
fi

# =============================================================================
# 7. OPTIMIZAR .GITIGNORE
# =============================================================================
log "📋 Verificando .gitignore..."

gitignore_entries=(
    "__pycache__/"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".pytest_cache/"
    "*.log"
    "*.tmp"
    "*.temp"
    "*.bak"
    "*.backup"
    ".DS_Store"
    "Thumbs.db"
    "temp_*"
    "simple_test_results/temp_*"
    "advanced_test_results/temp_*"
)

if [[ -f ".gitignore" ]]; then
    missing_entries=()
    for entry in "${gitignore_entries[@]}"; do
        if ! grep -q "^$entry$" .gitignore; then
            missing_entries+=("$entry")
        fi
    done
    
    if [[ ${#missing_entries[@]} -gt 0 ]]; then
        warning "Faltan ${#missing_entries[@]} entradas en .gitignore"
        log "Entradas faltantes: ${missing_entries[*]}"
        log "Considere añadirlas manualmente para evitar futuros archivos no deseados"
    else
        success ".gitignore está actualizado"
    fi
else
    warning "No se encontró archivo .gitignore"
fi

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================
echo ""
echo "============================================================================="
echo "📊 RESUMEN DE LIMPIEZA COMPLETADA"
echo "============================================================================="

# Calcular espacio liberado (aproximado)
log "Calculando estadísticas finales..."

# Contar archivos restantes
remaining_cache=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
remaining_pyc=$(find . -name "*.pyc" 2>/dev/null | wc -l)
remaining_pytest=$(find . -name ".pytest_cache" -type d 2>/dev/null | wc -l)

echo ""
success "✅ Limpieza completada exitosamente"
echo ""
echo "📈 Estadísticas:"
echo "   • Directorios __pycache__ restantes: $remaining_cache"
echo "   • Archivos .pyc restantes: $remaining_pyc"
echo "   • Directorios .pytest_cache restantes: $remaining_pytest"
echo ""

if [[ $remaining_cache -eq 0 ]] && [[ $remaining_pyc -eq 0 ]] && [[ $remaining_pytest -eq 0 ]]; then
    success "🎉 Repositorio completamente limpio"
else
    warning "⚠️  Algunos archivos cache pueden haber sido recreados durante la limpieza"
fi

echo ""
echo "🔄 Para mantener el repositorio limpio:"
echo "   1. Ejecute este script regularmente"
echo "   2. Configure su IDE para no generar archivos cache"
echo "   3. Use 'git clean -fdx' para limpieza más agresiva (¡CUIDADO!)"
echo ""
echo "============================================================================="
success "🧹 LIMPIEZA AUTOMÁTICA COMPLETADA"
echo "============================================================================="