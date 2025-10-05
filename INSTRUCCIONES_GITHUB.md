# Instrucciones para Crear Repositorio GitHub - SEACABAr

## 📋 Pasos para Crear el Repositorio en GitHub

### Opción 1: Usando la Interfaz Web de GitHub

1. **Ir a GitHub.com**
   - Navegar a https://github.com
   - Iniciar sesión en tu cuenta

2. **Crear Nuevo Repositorio**
   - Hacer clic en el botón "+" en la esquina superior derecha
   - Seleccionar "New repository"

3. **Configurar el Repositorio**
   - **Repository name**: `SEACABAr`
   - **Description**: `Sistema de Evaluación Automatizada de Cartuchos de Armas Balísticas - Forensic Ballistics Analysis System`
   - **Visibility**: Public ✅
   - **Initialize repository**: NO marcar ninguna opción (ya tenemos archivos locales)

4. **Crear el Repositorio**
   - Hacer clic en "Create repository"

### Opción 2: Instalar GitHub CLI (Recomendado)

```bash
# Instalar GitHub CLI
sudo apt install gh

# Autenticarse
gh auth login

# Crear repositorio
gh repo create SEACABAr --public --description "Sistema de Evaluación Automatizada de Cartuchos de Armas Balísticas - Forensic Ballistics Analysis System" --source=.
```

## 🔗 Conectar Repositorio Local con GitHub

Una vez creado el repositorio en GitHub, ejecutar estos comandos en la terminal:

```bash
# Navegar al directorio del proyecto
cd /home/marco/SEACABAr

# Agregar el repositorio remoto (reemplazar USERNAME con tu usuario de GitHub)
git remote add origin https://github.com/USERNAME/SEACABAr.git

# Verificar que el remoto se agregó correctamente
git remote -v

# Subir el código al repositorio
git push -u origin main
```

## 📁 Archivos que se Subirán

### ✅ Archivos Incluidos
- Todo el código fuente del proyecto
- Documentación (README.md, INFORME_ESTADO_PROYECTO.md)
- Archivos de configuración
- Tests y utilidades
- Dependencias (requirements.txt)

### ❌ Archivos Excluidos (por .gitignore)
- `uploads/` - Datos de muestra sensibles
- `venv_test/` - Entorno virtual de desarrollo
- `cache/` - Archivos temporales
- `data/` - Datos de usuario
- `database/ballistics.db*` - Base de datos local
- `config/backups/` - Respaldos de configuración
- Archivos de logs y temporales

## 🏷️ Configuración Adicional Recomendada

### Topics/Tags para el Repositorio
Agregar estos topics en GitHub para mejor descubrimiento:
- `forensics`
- `ballistics`
- `computer-vision`
- `image-processing`
- `pyqt5`
- `opencv`
- `machine-learning`
- `nist-standards`
- `forensic-science`
- `ballistic-analysis`

### Configurar Branch Protection (Opcional)
Para proyectos colaborativos:
1. Ir a Settings → Branches
2. Agregar regla para `main` branch
3. Configurar protecciones según necesidades

### Configurar Issues Templates (Opcional)
Crear templates para:
- Bug reports
- Feature requests
- Documentation improvements

## 📊 Verificación Post-Subida

Después de subir el código, verificar:

1. **Archivos principales presentes**:
   - ✅ README.md
   - ✅ INFORME_ESTADO_PROYECTO.md
   - ✅ main.py
   - ✅ requirements.txt
   - ✅ config.yaml

2. **Estructura de carpetas correcta**:
   - ✅ gui/
   - ✅ database/
   - ✅ matching/
   - ✅ core/
   - ✅ tests/

3. **Archivos sensibles excluidos**:
   - ❌ uploads/ (no debe aparecer)
   - ❌ venv_test/ (no debe aparecer)
   - ❌ *.db files (no deben aparecer)

## 🚀 Comandos de Referencia Rápida

```bash
# Estado del repositorio
git status

# Ver archivos que se subirán
git ls-files

# Ver archivos ignorados
git ls-files --others --ignored --exclude-standard

# Subir cambios futuros
git add .
git commit -m "Descripción del cambio"
git push origin main

# Clonar el repositorio (para otros usuarios)
git clone https://github.com/USERNAME/SEACABAr.git
```

## 📞 Soporte

Si encuentras problemas:
1. Verificar que Git esté configurado correctamente
2. Verificar conexión a internet
3. Verificar permisos de GitHub
4. Consultar documentación de Git/GitHub

---

**Nota**: Reemplazar `USERNAME` con tu nombre de usuario real de GitHub en todos los comandos.

*Instrucciones generadas para SEACABAr v2.0.0*