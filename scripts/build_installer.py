#!/usr/bin/env python3
"""
Script para construir el instalador de Windows para SEACABAr
Sistema de Análisis Balístico Automatizado
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build_dirs():
    """Limpia directorios de construcción anteriores"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Limpiado directorio: {dir_name}")

def create_spec_file():
    """Crea el archivo .spec para PyInstaller"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Archivos de datos adicionales
added_files = [
    ('config/unified_config/unified_config.yaml', '.'),
    ('gui/styles.py', 'gui'),
    ('database', 'database'),
    ('image_processing', 'image_processing'),
    ('matching', 'matching'),
    ('utils', 'utils'),
    ('deep_learning', 'deep_learning'),
]

# Módulos ocultos necesarios
hidden_imports = [
    'PyQt5.QtCore',
    'PyQt5.QtGui', 
    'PyQt5.QtWidgets',
    'cv2',
    'numpy',
    'sqlite3',
    'faiss',
    'sklearn',
    'PIL',
    'yaml',
    'logging',
    'threading',
    'queue',
    'json',
    'datetime',
    'pathlib',
    'gui.main_window',
    'gui.image_tab',
    'gui.comparison_tab',
    'gui.database_tab',
    'gui.reports_tab',
    'gui.visualization',
    'gui.styles',
    'database.vector_db',
    'image_processing.unified_preprocessor',
    'image_processing.unified_roi_detector',
    'image_processing.feature_extractor',
    'matching.unified_matcher',
    'utils.config',
    'utils.logger',
    'utils.validators',
    'deep_learning.ballistic_dl_models'
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SEACABAr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SEACABAr',
)
'''
    
    with open('SEACABAr.spec', 'w', encoding='utf-8') as f:
        f.write(spec_content)
    print("Archivo SEACABAr.spec creado")

def build_executable():
    """Construye el ejecutable usando PyInstaller"""
    try:
        print("Iniciando construcción del ejecutable...")
        cmd = [sys.executable, '-m', 'PyInstaller', '--clean', 'SEACABAr.spec']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Ejecutable construido exitosamente")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al construir ejecutable: {e}")
        print(f"Salida de error: {e.stderr}")
        return False

def create_installer_script():
    """Crea script NSIS para el instalador"""
    nsis_script = '''
; SEACABAr Installer Script
; Sistema de Análisis Balístico Automatizado

!define APPNAME "SEACABAr"
!define COMPANYNAME "Laboratorio de Balística Forense"
!define DESCRIPTION "Sistema de Análisis Balístico Automatizado"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0

!define HELPURL "https://github.com/seacabar/seacabar"
!define UPDATEURL "https://github.com/seacabar/seacabar/releases"
!define ABOUTURL "https://github.com/seacabar/seacabar"

!define INSTALLSIZE 500000

RequestExecutionLevel admin

InstallDir "$PROGRAMFILES\\${APPNAME}"

Name "${APPNAME}"
Icon "assets\\icon.ico"
outFile "SEACABAr_Installer.exe"

!include LogicLib.nsh

page components
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Se requieren privilegios de administrador!"
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "SEACABAr (requerido)"
    sectionIn RO
    setOutPath $INSTDIR
    
    ; Copiar archivos del programa
    file /r "dist\\SEACABAr\\*"
    
    ; Crear acceso directo en el escritorio
    createShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\SEACABAr.exe"
    
    ; Crear acceso directo en el menú inicio
    createDirectory "$SMPROGRAMS\\${APPNAME}"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\${APPNAME}.lnk" "$INSTDIR\\SEACABAr.exe"
    createShortCut "$SMPROGRAMS\\${APPNAME}\\Desinstalar.lnk" "$INSTDIR\\uninstall.exe"
    
    ; Registro de Windows
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayName" "${APPNAME} - ${DESCRIPTION}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "QuietUninstallString" "$INSTDIR\\uninstall.exe /S"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "InstallLocation" "$INSTDIR"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayIcon" "$INSTDIR\\SEACABAr.exe"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "Publisher" "${COMPANYNAME}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "HelpLink" "${HELPURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    writeRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoModify" 1
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "NoRepair" 1
    writeRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Crear desinstalador
    writeUninstaller "$INSTDIR\\uninstall.exe"
sectionEnd

section "Uninstall"
    ; Eliminar archivos
    rmDir /r "$INSTDIR"
    
    ; Eliminar accesos directos
    delete "$DESKTOP\\${APPNAME}.lnk"
    rmDir /r "$SMPROGRAMS\\${APPNAME}"
    
    ; Eliminar entradas del registro
    deleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APPNAME}"
sectionEnd
'''
    
    with open('installer.nsi', 'w', encoding='utf-8') as f:
        f.write(nsis_script)
    print("Script del instalador NSIS creado")

def create_assets_dir():
    """Crea directorio de assets con icono por defecto"""
    assets_dir = Path('assets')
    assets_dir.mkdir(exist_ok=True)
    
    # Crear un icono SVG simple si no existe
    if not (assets_dir / 'icon.ico').exists():
        svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <rect width="64" height="64" fill="#2E3440" rx="8"/>
  <circle cx="32" cy="20" r="8" fill="#5E81AC"/>
  <rect x="24" y="32" width="16" height="24" fill="#88C0D0" rx="2"/>
  <text x="32" y="50" text-anchor="middle" fill="#2E3440" font-family="Arial" font-size="12" font-weight="bold">SB</text>
</svg>'''
        with open(assets_dir / 'icon.svg', 'w') as f:
            f.write(svg_content)
        print("Icono SVG creado en assets/icon.svg")

def main():
    """Función principal del script de construcción"""
    print("=== SEACABAr - Constructor de Instalador ===")
    print("Sistema de Análisis Balístico Automatizado\n")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('main.py'):
        print("Error: No se encontró main.py. Ejecute desde el directorio raíz del proyecto.")
        sys.exit(1)
    
    # Limpiar construcciones anteriores
    clean_build_dirs()
    
    # Crear directorio de assets
    create_assets_dir()
    
    # Crear archivo .spec
    create_spec_file()
    
    # Construir ejecutable
    if build_executable():
        print("\n✓ Ejecutable construido exitosamente")
        
        # Crear script del instalador
        create_installer_script()
        print("✓ Script del instalador creado")
        
        print("\n=== Construcción Completada ===")
        print("Archivos generados:")
        print("- dist/SEACABAr/: Directorio con el ejecutable")
        print("- installer.nsi: Script para crear instalador NSIS")
        print("\nPara crear el instalador final:")
        print("1. Instale NSIS (Nullsoft Scriptable Install System)")
        print("2. Ejecute: makensis installer.nsi")
        
    else:
        print("\n✗ Error al construir el ejecutable")
        sys.exit(1)

if __name__ == "__main__":
    main()