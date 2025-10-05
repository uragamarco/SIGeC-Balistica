
; SIGeC-Balistica Installer Script
; Sistema Integral de Gestión Criminalística - Módulo Balístico

!define APPNAME "SIGeC-Balistica"
!define COMPANYNAME "Laboratorio de Balística Forense"
!define DESCRIPTION "Sistema Integral de Gestión Criminalística - Módulo Balístico"
!define VERSIONMAJOR 0
!define VERSIONMINOR 1
!define VERSIONBUILD 3

!define HELPURL "https://github.com/sigec-balistica/sigec-balistica"
!define UPDATEURL "https://github.com/sigec-balistica/sigec-balistica/releases"
!define ABOUTURL "https://github.com/sigec-balistica/sigec-balistica"

!define INSTALLSIZE 500000

RequestExecutionLevel admin

InstallDir "$PROGRAMFILES\${APPNAME}"

Name "${APPNAME}"
Icon "assets\icon.ico"
outFile "SIGeC-Balistica_Installer.exe"

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

section "SIGeC-Balistica (requerido)"
    sectionIn RO
    setOutPath $INSTDIR
    
    ; Copiar archivos del programa
    file /r "dist\SIGeC-Balistica\*"
    
    ; Crear acceso directo en el escritorio
    createShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\SIGeC-Balistica.exe"
    
    ; Crear acceso directo en el menú inicio
    createDirectory "$SMPROGRAMS\${APPNAME}"
    createShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\SIGeC-Balistica.exe"
    createShortCut "$SMPROGRAMS\${APPNAME}\Desinstalar.lnk" "$INSTDIR\uninstall.exe"
    
    ; Registro de Windows
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME} - ${DESCRIPTION}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$INSTDIR\uninstall.exe"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "QuietUninstallString" "$INSTDIR\uninstall.exe /S"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "InstallLocation" "$INSTDIR"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$INSTDIR\SIGeC-Balistica.exe"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "HelpLink" "${HELPURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoModify" 1
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoRepair" 1
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Crear desinstalador
    writeUninstaller "$INSTDIR\uninstall.exe"
sectionEnd

section "Uninstall"
    ; Eliminar archivos
    rmDir /r "$INSTDIR"
    
    ; Eliminar accesos directos
    delete "$DESKTOP\${APPNAME}.lnk"
    rmDir /r "$SMPROGRAMS\${APPNAME}"
    
    ; Eliminar entradas del registro
    deleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
sectionEnd
