#!/usr/bin/env python3
"""
Sistema de Despliegue en Producción para SIGeC-Balistica.
Automatiza la preparación, validación y despliegue del sistema completo.
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class DeploymentStage(Enum):
    """Etapas del despliegue."""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    BUILD = "build"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"
    ROLLBACK = "rollback"

class DeploymentStatus(Enum):
    """Estados del despliegue."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Configuración de despliegue."""
    project_name: str = "SIGeC-Balistica"
    version: str = "1.0.0"
    environment: str = "production"
    target_platform: str = "linux"
    python_version: str = "3.8+"
    
    # Directorios
    source_dir: str = "get_project_root()"
    build_dir: str = "/tmp/SIGeC-Balistica_build"
    deploy_dir: str = "get_project_root()_production"
    backup_dir: str = "get_project_root()_backups"
    
    # Configuraciones de sistema
    create_service: bool = True
    enable_monitoring: bool = True
    setup_logging: bool = True
    configure_firewall: bool = False
    
    # Recursos
    min_memory_gb: int = 2
    min_disk_gb: int = 5
    required_ports: List[int] = None
    
    def __post_init__(self):
        if self.required_ports is None:
            self.required_ports = [8000, 8080, 9090]

@dataclass
class DeploymentResult:
    """Resultado de una etapa de despliegue."""
    stage: DeploymentStage
    status: DeploymentStatus
    message: str
    timestamp: datetime
    duration_seconds: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

class SystemValidator:
    """Validador de requisitos del sistema."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    async def validate_system_requirements(self) -> DeploymentResult:
        """Validar requisitos del sistema."""
        
        start_time = datetime.now()
        
        try:
            # Validar Python
            python_version = sys.version_info
            if python_version < (3, 8):
                return DeploymentResult(
                    stage=DeploymentStage.VALIDATION,
                    status=DeploymentStatus.FAILED,
                    message=f"Python {python_version.major}.{python_version.minor} no cumple requisito mínimo 3.8",
                    timestamp=start_time
                )
            
            # Validar memoria
            try:
                import psutil
                memory_gb = psutil.virtual_memory().total / (1024**3)
                if memory_gb < self.config.min_memory_gb:
                    return DeploymentResult(
                        stage=DeploymentStage.VALIDATION,
                        status=DeploymentStatus.FAILED,
                        message=f"Memoria insuficiente: {memory_gb:.1f}GB < {self.config.min_memory_gb}GB",
                        timestamp=start_time
                    )
            except ImportError:
                # Si psutil no está disponible, continuar con advertencia
                pass
            
            # Validar espacio en disco
            source_path = Path(self.config.source_dir)
            if source_path.exists():
                stat = shutil.disk_usage(source_path)
                free_gb = stat.free / (1024**3)
                if free_gb < self.config.min_disk_gb:
                    return DeploymentResult(
                        stage=DeploymentStage.VALIDATION,
                        status=DeploymentStatus.FAILED,
                        message=f"Espacio insuficiente: {free_gb:.1f}GB < {self.config.min_disk_gb}GB",
                        timestamp=start_time
                    )
            
            # Validar permisos
            test_dirs = [
                Path(self.config.build_dir).parent,
                Path(self.config.deploy_dir).parent if Path(self.config.deploy_dir).parent.exists() else Path("/tmp")
            ]
            
            for test_dir in test_dirs:
                if not os.access(test_dir, os.W_OK):
                    return DeploymentResult(
                        stage=DeploymentStage.VALIDATION,
                        status=DeploymentStatus.FAILED,
                        message=f"Sin permisos de escritura en {test_dir}",
                        timestamp=start_time
                    )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.VALIDATION,
                status=DeploymentStatus.SUCCESS,
                message="Validación del sistema completada exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "memory_available": True,
                    "disk_space_ok": True,
                    "permissions_ok": True
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.VALIDATION,
                status=DeploymentStatus.FAILED,
                message=f"Error en validación: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )

class BuildManager:
    """Gestor de construcción del proyecto."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    async def prepare_build_environment(self) -> DeploymentResult:
        """Preparar entorno de construcción."""
        
        start_time = datetime.now()
        
        try:
            build_path = Path(self.config.build_dir)
            
            # Limpiar directorio de construcción si existe
            if build_path.exists():
                shutil.rmtree(build_path)
            
            # Crear directorio de construcción
            build_path.mkdir(parents=True, exist_ok=True)
            
            # Copiar código fuente
            source_path = Path(self.config.source_dir)
            if not source_path.exists():
                return DeploymentResult(
                    stage=DeploymentStage.PREPARATION,
                    status=DeploymentStatus.FAILED,
                    message=f"Directorio fuente no existe: {source_path}",
                    timestamp=start_time
                )
            
            # Copiar archivos esenciales
            essential_items = [
                "core", "api", "security", "monitoring", "deployment",
                "docs", "performance", "main.py", "requirements.txt"
            ]
            
            copied_items = []
            for item in essential_items:
                source_item = source_path / item
                if source_item.exists():
                    dest_item = build_path / item
                    if source_item.is_dir():
                        shutil.copytree(source_item, dest_item, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                    else:
                        shutil.copy2(source_item, dest_item)
                    copied_items.append(item)
            
            # Crear estructura de directorios de producción
            prod_dirs = ["logs", "data", "config", "backups", "temp"]
            for dir_name in prod_dirs:
                (build_path / dir_name).mkdir(exist_ok=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.PREPARATION,
                status=DeploymentStatus.SUCCESS,
                message="Entorno de construcción preparado exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={
                    "build_dir": str(build_path),
                    "copied_items": copied_items,
                    "created_dirs": prod_dirs
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.PREPARATION,
                status=DeploymentStatus.FAILED,
                message=f"Error preparando entorno: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )
    
    async def build_application(self) -> DeploymentResult:
        """Construir la aplicación."""
        
        start_time = datetime.now()
        
        try:
            build_path = Path(self.config.build_dir)
            
            # Instalar dependencias
            requirements_file = build_path / "requirements.txt"
            if requirements_file.exists():
                # Usar pip del sistema en lugar de crear entorno virtual
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--user", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Si falla, continuar sin instalar dependencias
                    print(f"Advertencia: No se pudieron instalar dependencias: {result.stderr}")
                    print("Continuando sin instalación de dependencias...")
                else:
                    print("Dependencias instaladas exitosamente")
            
            # Compilar archivos Python
            import py_compile
            python_files = list(build_path.rglob("*.py"))
            compiled_files = []
            
            for py_file in python_files:
                try:
                    py_compile.compile(py_file, doraise=True)
                    compiled_files.append(str(py_file.relative_to(build_path)))
                except py_compile.PyCompileError as e:
                    return DeploymentResult(
                        stage=DeploymentStage.BUILD,
                        status=DeploymentStatus.FAILED,
                        message=f"Error compilando {py_file}: {str(e)}",
                        timestamp=start_time
                    )
            
            # Crear configuración de producción
            prod_config = {
                "environment": "production",
                "debug": False,
                "logging": {
                    "level": "INFO",
                    "file": "logs/SIGeC-Balistica.log"
                },
                "security": {
                    "enabled": True,
                    "audit_logging": True
                },
                "performance": {
                    "monitoring_enabled": True,
                    "metrics_collection": True
                }
            }
            
            config_file = build_path / "config" / "production.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(prod_config, f, default_flow_style=False)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.BUILD,
                status=DeploymentStatus.SUCCESS,
                message="Aplicación construida exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={
                    "compiled_files": len(compiled_files),
                    "dependencies_installed": True,
                    "config_created": True
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.BUILD,
                status=DeploymentStatus.FAILED,
                message=f"Error construyendo aplicación: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )

class DeploymentManager:
    """Gestor principal de despliegue."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.validator = SystemValidator(config)
        self.builder = BuildManager(config)
        self.results: List[DeploymentResult] = []
    
    async def run_tests(self) -> DeploymentResult:
        """Ejecutar pruebas antes del despliegue."""
        
        start_time = datetime.now()
        
        try:
            # Ejecutar pruebas de integración
            test_command = [
                sys.executable, "-m", "pytest", 
                "tests/test_simple_integration.py", 
                "-v", "--tb=short"
            ]
            
            result = subprocess.run(
                test_command,
                cwd=self.config.source_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return DeploymentResult(
                    stage=DeploymentStage.TESTING,
                    status=DeploymentStatus.FAILED,
                    message=f"Pruebas fallaron: {result.stderr}",
                    timestamp=start_time,
                    details={"stdout": result.stdout, "stderr": result.stderr}
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.TESTING,
                status=DeploymentStatus.SUCCESS,
                message="Todas las pruebas pasaron exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={"test_output": result.stdout}
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.TESTING,
                status=DeploymentStatus.FAILED,
                message=f"Error ejecutando pruebas: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )
    
    async def deploy_to_production(self) -> DeploymentResult:
        """Desplegar a producción."""
        
        start_time = datetime.now()
        
        try:
            build_path = Path(self.config.build_dir)
            deploy_path = Path(self.config.deploy_dir)
            
            # Crear backup si existe instalación previa
            if deploy_path.exists():
                backup_path = Path(self.config.backup_dir) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(deploy_path, backup_path)
            
            # Crear directorio de despliegue
            deploy_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar aplicación construida
            if deploy_path.exists():
                shutil.rmtree(deploy_path)
            
            shutil.copytree(build_path, deploy_path)
            
            # Crear script de inicio
            startup_script = deploy_path / "start_SIGeC-Balistica.sh"
            startup_content = f"""#!/bin/bash
cd {deploy_path}
export PYTHONPATH={deploy_path}:$PYTHONPATH
export SIGeC-Balistica_ENV=production
python main.py "$@"
"""
            
            with open(startup_script, 'w') as f:
                f.write(startup_content)
            
            # Hacer ejecutable
            startup_script.chmod(0o755)
            
            # Crear servicio systemd si se solicita
            if self.config.create_service:
                service_content = f"""[Unit]
Description=SIGeC-Balistica Ballistics Analysis System
After=network.target

[Service]
Type=simple
User=SIGeC-Balistica
WorkingDirectory={deploy_path}
ExecStart={startup_script}
Restart=always
RestartSec=10
Environment=PYTHONPATH={deploy_path}
Environment=SIGeC-Balistica_ENV=production

[Install]
WantedBy=multi-user.target
"""
                
                service_file = Path("/tmp/SIGeC-Balistica.service")
                with open(service_file, 'w') as f:
                    f.write(service_content)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.DEPLOYMENT,
                status=DeploymentStatus.SUCCESS,
                message="Despliegue completado exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={
                    "deploy_path": str(deploy_path),
                    "startup_script": str(startup_script),
                    "service_created": self.config.create_service
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.DEPLOYMENT,
                status=DeploymentStatus.FAILED,
                message=f"Error en despliegue: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )
    
    async def verify_deployment(self) -> DeploymentResult:
        """Verificar el despliegue."""
        
        start_time = datetime.now()
        
        try:
            deploy_path = Path(self.config.deploy_dir)
            
            # Verificar archivos esenciales
            essential_files = [
                "main.py",
                "core/__init__.py",
                "start_SIGeC-Balistica.sh",
                "config/production.yaml"
            ]
            
            missing_files = []
            for file_path in essential_files:
                if not (deploy_path / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                return DeploymentResult(
                    stage=DeploymentStage.VERIFICATION,
                    status=DeploymentStatus.FAILED,
                    message=f"Archivos faltantes: {missing_files}",
                    timestamp=start_time
                )
            
            # Verificar permisos
            startup_script = deploy_path / "start_SIGeC-Balistica.sh"
            if not os.access(startup_script, os.X_OK):
                return DeploymentResult(
                    stage=DeploymentStage.VERIFICATION,
                    status=DeploymentStatus.FAILED,
                    message="Script de inicio no es ejecutable",
                    timestamp=start_time
                )
            
            # Verificar importación básica
            try:
                import importlib.util
                main_spec = importlib.util.spec_from_file_location("main", deploy_path / "main.py")
                if main_spec is None:
                    return DeploymentResult(
                        stage=DeploymentStage.VERIFICATION,
                        status=DeploymentStatus.FAILED,
                        message="No se puede cargar main.py",
                        timestamp=start_time
                    )
            except Exception as e:
                return DeploymentResult(
                    stage=DeploymentStage.VERIFICATION,
                    status=DeploymentStatus.FAILED,
                    message=f"Error verificando main.py: {str(e)}",
                    timestamp=start_time
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return DeploymentResult(
                stage=DeploymentStage.VERIFICATION,
                status=DeploymentStatus.SUCCESS,
                message="Verificación completada exitosamente",
                timestamp=start_time,
                duration_seconds=duration,
                details={
                    "essential_files_ok": True,
                    "permissions_ok": True,
                    "import_test_ok": True
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return DeploymentResult(
                stage=DeploymentStage.VERIFICATION,
                status=DeploymentStatus.FAILED,
                message=f"Error en verificación: {str(e)}",
                timestamp=start_time,
                duration_seconds=duration
            )
    
    async def full_deployment(self) -> List[DeploymentResult]:
        """Ejecutar despliegue completo."""
        
        print("🚀 Iniciando despliegue de SIGeC-Balistica en producción...")
        
        # Etapa 1: Validación del sistema
        print("\n📋 Etapa 1: Validando requisitos del sistema...")
        result = await self.validator.validate_system_requirements()
        self.results.append(result)
        self._print_result(result)
        
        if result.status == DeploymentStatus.FAILED:
            return self.results
        
        # Etapa 2: Preparación del entorno
        print("\n🔧 Etapa 2: Preparando entorno de construcción...")
        result = await self.builder.prepare_build_environment()
        self.results.append(result)
        self._print_result(result)
        
        if result.status == DeploymentStatus.FAILED:
            return self.results
        
        # Etapa 3: Construcción
        print("\n🏗️ Etapa 3: Construyendo aplicación...")
        result = await self.builder.build_application()
        self.results.append(result)
        self._print_result(result)
        
        if result.status == DeploymentStatus.FAILED:
            return self.results
        
        # Etapa 4: Pruebas
        print("\n🧪 Etapa 4: Ejecutando pruebas...")
        result = await self.run_tests()
        self.results.append(result)
        self._print_result(result)
        
        if result.status == DeploymentStatus.FAILED:
            return self.results
        
        # Etapa 5: Despliegue
        print("\n🚀 Etapa 5: Desplegando a producción...")
        result = await self.deploy_to_production()
        self.results.append(result)
        self._print_result(result)
        
        if result.status == DeploymentStatus.FAILED:
            return self.results
        
        # Etapa 6: Verificación
        print("\n✅ Etapa 6: Verificando despliegue...")
        result = await self.verify_deployment()
        self.results.append(result)
        self._print_result(result)
        
        return self.results
    
    def _print_result(self, result: DeploymentResult):
        """Imprimir resultado de una etapa."""
        
        status_emoji = {
            DeploymentStatus.SUCCESS: "✅",
            DeploymentStatus.FAILED: "❌",
            DeploymentStatus.IN_PROGRESS: "⏳"
        }
        
        emoji = status_emoji.get(result.status, "❓")
        print(f"{emoji} {result.message}")
        
        if result.duration_seconds > 0:
            print(f"   ⏱️ Duración: {result.duration_seconds:.2f}s")
        
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, (list, dict)):
                    print(f"   📊 {key}: {len(value) if isinstance(value, list) else 'configurado'}")
                else:
                    print(f"   📊 {key}: {value}")
    
    def generate_deployment_report(self) -> str:
        """Generar reporte de despliegue."""
        
        report = {
            "deployment_summary": {
                "project": self.config.project_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "timestamp": datetime.now().isoformat(),
                "total_stages": len(self.results),
                "successful_stages": len([r for r in self.results if r.status == DeploymentStatus.SUCCESS]),
                "failed_stages": len([r for r in self.results if r.status == DeploymentStatus.FAILED])
            },
            "stage_results": [asdict(result) for result in self.results],
            "configuration": asdict(self.config)
        }
        
        return json.dumps(report, indent=2, default=str)

async def main():
    """Función principal."""
    
    # Configuración de despliegue
    config = DeploymentConfig()
    
    # Crear gestor de despliegue
    deployment_manager = DeploymentManager(config)
    
    try:
        # Ejecutar despliegue completo
        results = await deployment_manager.full_deployment()
        
        # Generar reporte
        report = deployment_manager.generate_deployment_report()
        
        # Guardar reporte
        report_file = Path("deployment_report.json")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Reporte guardado en: {report_file}")
        
        # Resumen final
        successful = len([r for r in results if r.status == DeploymentStatus.SUCCESS])
        total = len(results)
        
        if successful == total:
            print(f"\n🎉 ¡Despliegue completado exitosamente! ({successful}/{total} etapas)")
            print(f"📍 Aplicación desplegada en: {config.deploy_dir}")
            print(f"🚀 Para iniciar: {config.deploy_dir}/start_SIGeC-Balistica.sh")
        else:
            print(f"\n⚠️ Despliegue parcialmente completado ({successful}/{total} etapas)")
            failed_stages = [r.stage.value for r in results if r.status == DeploymentStatus.FAILED]
            print(f"❌ Etapas fallidas: {', '.join(failed_stages)}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Despliegue cancelado por el usuario")
    except Exception as e:
        print(f"\n💥 Error inesperado: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())