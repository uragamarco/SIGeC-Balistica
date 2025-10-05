# Instrucciones para Crear Repositorio GitHub - SIGeC-Balistica

## Información del Proyecto
- **Nombre**: SIGeC-Balistica
- **Descripción**: Sistema Integral de Gestión Criminalística - Módulo Balístico
- **Versión**: v0.1.3
- **Tipo**: Repositorio público

## Opción 1: Crear Repositorio desde la Interfaz Web de GitHub

### Pasos:
1. **Acceder a GitHub**
   - Ir a https://github.com
   - Iniciar sesión en tu cuenta

2. **Crear Nuevo Repositorio**
   - Hacer clic en el botón "+" en la esquina superior derecha
   - Seleccionar "New repository"

3. **Configurar el Repositorio**
   - **Repository name**: `SIGeC-Balistica`
   - **Description**: `Sistema Integral de Gestión Criminalística - Módulo Balístico v0.1.3`
   - **Visibility**: Public ✅
   - **Initialize repository**: NO marcar ninguna opción (ya tenemos archivos locales)

4. **Crear el Repositorio**
   - Hacer clic en "Create repository"

## Opción 2: Crear Repositorio usando GitHub CLI

### Prerrequisitos:
```bash
# Instalar GitHub CLI (si no está instalado)
# Ubuntu/Debian:
sudo apt update
sudo apt install gh

# O usando snap:
sudo snap install gh

# Autenticarse
gh auth login
```

### Comando para crear el repositorio:
```bash
gh repo create SIGeC-Balistica --public --description "Sistema Integral de Gestión Criminalística - Módulo Balístico v0.1.3"
```

## Conectar Repositorio Local con GitHub

### Una vez creado el repositorio en GitHub, ejecutar:

```bash
# Agregar el remote origin
git remote add origin https://github.com/TU_USUARIO/SIGeC-Balistica.git

# Verificar que el remote se agregó correctamente
git remote -v

# Hacer push de la rama principal
git branch -M main
git push -u origin main
```

## Archivos Incluidos en el Repositorio

### Archivos Principales:
- `README.md` - Documentación principal del proyecto
- `main.py` - Archivo principal de la aplicación
- `requirements.txt` - Dependencias de Python
- `config.yaml` - Configuración principal
- `pytest.ini` - Configuración de pruebas

### Documentación:
- `INFORME_ESTADO_PROYECTO.md` - Estado actual del desarrollo
- `DOCS/` - Carpeta con documentación técnica completa
- `DOCS/README.md` - Documentación técnica
- `DOCS/deployment_summary.md` - Resumen de despliegue
- `DOCS/plan_desarrollo_seacabar.md` - Plan de desarrollo

### Código Fuente:
- `gui/` - Interfaz gráfica de usuario
- `core/` - Núcleo del sistema
- `image_processing/` - Procesamiento de imágenes
- `matching/` - Algoritmos de comparación
- `nist_standards/` - Estándares NIST
- `database/` - Sistema de base de datos
- `deep_learning/` - Modelos de aprendizaje profundo
- `common/` - Utilidades comunes
- `utils/` - Herramientas auxiliares

### Archivos de Configuración:
- `config/` - Configuraciones del sistema
- `scripts/` - Scripts de utilidad
- `tests/` - Suite de pruebas

### Archivos Excluidos (.gitignore):
- `__pycache__/` - Cache de Python
- `*.pyc` - Archivos compilados de Python
- `.pytest_cache/` - Cache de pytest
- `logs/` - Archivos de log
- `temp/` - Archivos temporales
- `.env` - Variables de entorno
- `INSTRUCCIONES_GITHUB.md` - Este archivo de instrucciones

## Configuraciones Adicionales Recomendadas

### 1. Configurar Topics (Etiquetas)
En la página del repositorio en GitHub:
- Ir a "Settings" → "General"
- En la sección "Topics", agregar:
  - `balistica`
  - `forense`
  - `criminalistica`
  - `python`
  - `opencv`
  - `nist-standards`
  - `image-processing`
  - `deep-learning`

### 2. Configurar Branch Protection
- Ir a "Settings" → "Branches"
- Agregar regla para la rama `main`:
  - ✅ Require pull request reviews before merging
  - ✅ Require status checks to pass before merging

### 3. Configurar Issues Templates
- Crear templates para:
  - Bug reports
  - Feature requests
  - Documentation improvements

## Verificación Post-Subida

### Verificar que todo se subió correctamente:
```bash
# Verificar el estado del repositorio
git status

# Ver el historial de commits
git log --oneline -10

# Verificar archivos remotos
git ls-remote origin
```

### Verificar en GitHub:
1. **Archivos**: Confirmar que todos los archivos están presentes
2. **README**: Verificar que se muestra correctamente en la página principal
3. **Releases**: Considerar crear un release para v0.1.3
4. **Issues**: Verificar que están habilitados
5. **Wiki**: Considerar habilitar para documentación adicional

## Comandos de Referencia Rápida

```bash
# Clonar el repositorio (para otros desarrolladores)
git clone https://github.com/TU_USUARIO/SIGeC-Balistica.git

# Actualizar repositorio local
git pull origin main

# Agregar cambios y hacer commit
git add .
git commit -m "Descripción del cambio"
git push origin main

# Crear y cambiar a nueva rama
git checkout -b nueva-funcionalidad
git push -u origin nueva-funcionalidad

# Ver estado del repositorio
git status
git log --oneline -5
```

## Información de Soporte

- **Documentación**: Ver carpeta `DOCS/` para documentación técnica completa
- **Issues**: Usar el sistema de issues de GitHub para reportar problemas
- **Contribuciones**: Ver `README.md` para guías de contribución
- **Licencia**: Verificar archivo de licencia en el repositorio

## Notas Importantes

1. **Seguridad**: No incluir credenciales, API keys o información sensible
2. **Tamaño**: El repositorio actual es aproximadamente 50MB
3. **Compatibilidad**: Compatible con Python 3.8+
4. **Dependencias**: Ver `requirements.txt` para lista completa
5. **Testing**: Ejecutar `pytest` antes de hacer push

---

**SIGeC-Balistica v0.1.3** - Sistema Integral de Gestión Criminalística - Módulo Balístico